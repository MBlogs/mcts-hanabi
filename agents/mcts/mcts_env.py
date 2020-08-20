import pyhanabi
from rl_env import HanabiEnv
from agents.mcts.mcts_determinizer import MCTSDeterminizer
from pyhanabi import HanabiMove
from pyhanabi import HanabiEndOfGameType
import random


class MCTSEnv(HanabiEnv):
  def __init__(self, config):
    # This is the forward model for a particular MCTS player. Note it's position
    self.mcts_player = config['mcts_player']
    self.remember_hand = None
    self.determiniser = MCTSDeterminizer()
    super().__init__(config)

  def reset(self, observations):
    self.record_moves.reset(observations)

  def step(self, action):
    debug = False

    # Convert action into HanabiMove
    if isinstance(action, dict):
      move = self._build_move(action)
    elif isinstance(action, int):
      move = self.game.get_move(action)
    elif isinstance(action, pyhanabi.HanabiMove):
      move = action
    else:
      raise ValueError("Expected action as dict or int, got: {}".format(
          action))

    # If Play or Discard, make note of the card
    actioned_card = None
    action_player = self.state.cur_player()
    if move.type() == pyhanabi.HanabiMoveType.DISCARD or move.type() == pyhanabi.HanabiMoveType.PLAY:
      actioned_card = self.state.player_hands()[self.state.cur_player()][move.card_index()]

    # Apply the move
    if debug: print(f"MB: mcts_env.step: Player {self.state.cur_player()} applying move {move}")
    self.state.apply_move(move)

    # If cur_player is now chance, player needs a random card dealt (KEEP)
    while self.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
      self.state.deal_random_card()
      if debug: print(f"mcts_env.step: Player {action_player} dealt random card")

    # If the acting player was not me, restore as best as possible their hand
    if action_player != self.mcts_player:
     self.restore_hand(action_player, self.remember_hand, actioned_card, move.card_index())
     if debug: print(f"mcts_env.step: Player {action_player} restored hand")

    # Now we're onto the  next player. If not me, remember, then replace their hand
    if self.state.cur_player() != self.mcts_player:
      if debug: print(f"mcts_env.step: Player {self.state.cur_player()} saving hand")
      self.remember_hand = self.state.player_hands()[self.state.cur_player()]
      self.replace_hand(self.state.cur_player())
      if debug: print(f"mcts_env.step: Player {self.state.cur_player()} replaced hand")

    if debug: self.print_state()
    # Now make observation, as action player hand is restored and new player hand is redeterminised
    observations = self._make_observation_all_players()
    # Now move complete, update the stats record
    self.record_moves.update(move, observations["player_observations"][action_player], action_player, 0)
    reward = self.reward()
    done = self.state.is_terminal()
    info = {}
    return (observations, reward, done, info)

  def game_stats(self):
    return self.record_moves.game_stats

  def player_stats(self):
    return self.record_moves.player_stats

  def reward(self):
    """Custom reward function for use during RIS-MCTS rollouts
    This is therefore not the same as the overall game score
    """
    score = self.fireworks_score() - self.record_moves.regret()
    return score

  def return_hand(self,player):
    """Return all cards from a player's hand to the deck
    Note: a return move retains card knowledge in each spot"""
    debug = False
    hand_size = len(self.state.player_hands()[player])
    for card_index in range(hand_size):
      # Return the card in furthest left position (cards always shift downwards)
      return_move = HanabiMove.get_return_move(card_index=0, player=player)
      self.state.apply_move(return_move)
    if debug: print(f"mcts_env.restore_hand: Player {player} returned hand")


  def valid_hand(self, player, original_hand_size, card_knowledge):
    """Find a valid full hand"""
    debug = False
    replacement_hand = []
    if debug: print(f"mcts_env.valid_hand: Player {player} finding a valid replacement hand")

    # Try to find a valid replacement hand. While loop is needed because sometimes there is intra-hand conflict
    while len(replacement_hand) < original_hand_size:
      replacement_hand = []
      for card_index in range(original_hand_size):
        # Make sure to remove the current replacement cards from potential valid card pool
        valid_card = self.determiniser.valid_card(player, card_index, self.state.player_hands(),
                                                  self.state.discard_pile(), self.state.fireworks()
                                                  , card_knowledge, replacement_hand)
        # If there are no valid_cards, restart. Else, append to replacement_hand
        if not valid_card:
          if debug: print(f"mcts_env.valid_hand: Player {player} found broken replacement hand. Restarting.")
          break
        replacement_hand.append(valid_card)

    if debug: print(f"mcts_env.valid_hand: Player {player} found valid replacement hand {replacement_hand}")
    return replacement_hand


  def replace_hand(self, player):
    """ Replace a player hand with a different valid one
    Note: card_knowledge is retained"""
    debug = False
    hand_size = len(self.state.player_hands()[player])
    if debug: hand = self.state.player_hands()[player]
    temp_observation = self.state.observation(player)
    card_knowledge = temp_observation.card_knowledge()[0]

    # First put all cards from hand back in deck to become possibilities
    self.return_hand(player)

    # Then find a valid replacement hand that is consistent with all the facts so far
    replacement_hand = self.valid_hand(player, hand_size, card_knowledge)
    for card_index in range(len(replacement_hand)):
      card = replacement_hand[card_index]
      deal_specific_move = HanabiMove.get_deal_specific_move(card_index, player, card.color(), card.rank())
      self.state.apply_move(deal_specific_move)
      if debug: print(f"mcts_env.replace_hand: Player {player} replaced card {hand[card_index]} with {card}")

  def restore_hand(self, player, remember_hand, removed_card=None, removed_card_index = -1):
    """As best as possible, restore player's current hand as closely as possible to a remembered one
    remember_hand: A hand to match (usually remembered before a redeterminisation)
    removed_card: This card was played or discarded. Used to resolve intra-hand conflict.
    removed_card_index: This card was played or discarded. Used to skip over it.
    """
    debug = False
    # Source card knowledge upfront
    temp_observation = self.state.observation(player)
    card_knowledge = temp_observation.card_knowledge()[0]
    # Note size of hand in case we need to deal a random
    hand_size = len(self.state.player_hands()[player])

    # Start by returning all cards to deck
    self.return_hand(player)

    card_index = 0

    for remember_card_index in range(len(remember_hand)):
      # Note: Have to iterate over remembered hand here; could have gone 5 > 4
      # Reember card index iterates over remembered hand
      # card_index iterates over restored hand

      # Skip over the remembered card if it was the one played.
      if remember_card_index == removed_card_index:
        continue

      # Assign card in remembered hand to restore
      card = remember_hand[remember_card_index]

      # When Remembered == Actioned card, we need to check we're not adding too many of this card
      if removed_card and card == removed_card:
        # Additional cards will mean there are no intrahand conflict.
        # Make sure to not double count a card that is now in discard pile
        additional_cards = [remember_hand[i] for i in range(remember_card_index + 1, len(remember_hand)) if
                            i != removed_card_index]
        if debug: print(f"mcts_env.restore_hand: Player {player} played {card} which previously had. Checking validity")
        valid_cards = self.determiniser.valid_cards(player, card_index, self.state.player_hands()
                                                    , self.state.discard_pile(), self.state.fireworks(), card_knowledge
                                                    , additional_cards)
        # If card is no longer valid, replace with random valid
        if not any(c == card for c in valid_cards):
          if debug: print(f"mcts_env.restore_hand: Player {player} card {card} no longer valid. Replace valid")
          if len(valid_cards) > 0:
            card = random.choice(valid_cards)
          else:
            # Technically there could be no more valid cards (as the only other one is later in the hand)
            # Here we invoke a variant of deal specific that destroys card knowledge
            if debug: print(f"mcts_env.restore_hand: Player {player} card {card} no longer valid and no other valid!")
            self.state.remove_knowledge(player, card_index)
            if debug: print(f"mcts_env.restore_hand: Player {player} removed knowledge for card {card}")
            # Perform same check but without card knowledge
            card = self.determiniser.valid_card(player, card_index, self.state.player_hands()
                                                 , self.state.discard_pile(), self.state.fireworks(), None
                                                 , additional_cards)

      # Now we're happy and can deal the identified card
      deal_specific_move = HanabiMove.get_deal_specific_move(card_index, player, card.color(), card.rank())
      self.state.apply_move(deal_specific_move)
      if debug: print(f"mcts_env.restore_hand: Player {player} hand now {self.state.player_hands()[player]}")
      card_index += 1

    # Double check the hand is of right size. If not, deal the final card
    if hand_size > len(self.state.player_hands()[player]):
      card = self.determiniser.valid_card(player, card_index, self.state.player_hands()
                                                  , self.state.discard_pile(), self.state.fireworks(), card_knowledge)
      deal_specific_move = HanabiMove.get_deal_specific_move(card_index, player, card.color(), card.rank())
      if debug: print(f"mcts_env.restore_hand: Player {player} restoring card {card}")
      self.state.apply_move(deal_specific_move)
      if debug: print(f"mcts_env.restore_hand: Player {player} hand now {self.state.player_hands()[player]}")


def make(environment_name="Hanabi-Full", num_players=2, mcts_player=0, pyhanabi_path=None):
  """Make an environment.

  Args:
    environment_name: str, Name of the environment to instantiate.
    num_players: int, Number of players in this game.
    pyhanabi_path: str, absolute path to header files for c code linkage.

  Returns:
    env: An `Environment` object.

  Raises:
    ValueError: Unknown environment name.
  """

  if pyhanabi_path is not None:
    prefixes=(pyhanabi_path,)
    assert pyhanabi.try_cdef(prefixes=prefixes), "cdef failed to load"
    assert pyhanabi.try_load(prefixes=prefixes), "library failed to load"

  if (environment_name == "Hanabi-Full" or
      environment_name == "Hanabi-Full-CardKnowledge"):
    return MCTSEnv(
        config={
            "colors":
                5,
            "ranks":
                5,
            "players":
                num_players,
            "mcts_player":
                mcts_player,
            "max_information_tokens":
                8,
            "max_life_tokens":
                3,
            "observation_type":
                pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
        })
  else:
    raise ValueError("Unknown environment {}".format(environment_name))