import pyhanabi
from rl_env import HanabiEnv
from agents.mcts.mcts_determinizer import MCTSDeterminizer
from pyhanabi import HanabiMove
import random


class MCTSEnv(HanabiEnv):
  def __init__(self, config):
    # This is the forward model for a particular MCTS player. Note it's position
    self.mcts_player = config['mcts_player']
    self.remember_hand = None
    self.determiniser = MCTSDeterminizer()
    super().__init__(config)


  def step(self, action):
    # ToDo: Handle redeterminizing before first step called (1st case)
    # Answer: That will never happen?
    debug = True

    # Convert action into HanabiMove
    if isinstance(action, dict):
      action = self._build_move(action)
    elif isinstance(action, int):
      action = self.game.get_move(action)
    elif isinstance(action, pyhanabi.HanabiMove):
      pass
    else:
      raise ValueError("Expected action as dict or int, got: {}".format(
          action))

    # If Play or Discard, make note of the card
    actioned_card = None
    action_player = self.state.cur_player()
    if action.type() == pyhanabi.HanabiMoveType.DISCARD or action.type() == pyhanabi.HanabiMoveType.PLAY:
      actioned_card = self.state.player_hands()[self.state.cur_player()][action.card_index()]

    # Apply the action
    if debug: print(f"MB: mcts_env.step: Player {self.state.cur_player()} applying action {action}")
    self.state.apply_move(action)

    # If cur_player is now chance, player needs a random card dealt (KEEP)
    while self.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
      self.state.deal_random_card()
      if debug: print(f"mcts_env.step: Player {action_player} dealt random card")

    # If the acting player was not me, restore as best as possible their hand
    if action_player != self.mcts_player:
     self.restore_hand(action_player, self.remember_hand, actioned_card, action.card_index())
     if debug: print(f"mcts_env.step: Player {action_player} restored hand")

    # Now we're onto the  next player. If not me, remember then replace their hand
    if self.state.cur_player() != self.mcts_player:
      if debug: print(f"mcts_env.step: Player {self.state.cur_player()} saving hand")
      self.remember_hand = self.state.player_hands()[self.state.cur_player()]
      self.replace_hand(self.state.cur_player())
      if debug: print(f"mcts_env.step: Player {self.state.cur_player()} replaced hand")

    # Now make observation, as action player hand is restored and new player hand is redeterminised
    observation = self._make_observation_all_players()
    if debug: self.print_state()

    reward = self.state.reward()
    done = self.state.is_terminal()
    info = {}
    return (observation, reward, done, info)


  def restore_hand(self, player, remember_hand, removed_card=None, removed_card_index = -1):
    """As best as possible, restore current player hand to the one passed in
    remember_hand: Their hand before it was replaced on their turn
    removed_card_index: This card was played or discarded. Used to skip over it.
    """
    debug = True
    # Source card knowledge upfront
    temp_observation = self.state.observation(player)
    card_knowledge = temp_observation.card_knowledge()[0]
    # Note size of hand in case we need to deal a random
    hand_size = len(self.state.player_hands()[player])
    destroy_knowledge = False

    # Start by returning all cards to deck (resolves intra-hand conflict)
    for card_index in range(hand_size):
      # Return the card in their current position (return will always be the oldest card)
      return_move = HanabiMove.get_return_move(card_index=0, player=player)
      if debug: print(f"mcts_env.restore_hand: Player {player} returning card: {self.state.player_hands()[player][0]}")
      self.state.apply_move(return_move)
      if debug: print(f"mcts_env.restore_hand: Player {player} hand now {self.state.player_hands()[player]}")

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
        if debug: print(f"mcts_env.restore_hand: Player {player} played {card} which previously had. Checking validity")
        valid_cards = self.determiniser.valid_cards(player, card_index, self.state.player_hands()
                                                    , self.state.discard_pile(), self.state.fireworks(), card_knowledge
                                                    , remember_hand[card_index+1:len(remember_hand)])
        # If card is no longer valid, replace with random valid
        if not any(c == card for c in valid_cards):
          if len(valid_cards) > 0:
            if debug: print(f"mcts_env.restore_hand: Player {player} card {card} no longer valid. Replace with random other valid card")
            card = random.choice(valid_cards)
          else:
            # Technically there could be no more valid cards (as the only other one is later in the hand)
            # Here we invoke a variant of deal specific that destroys card knowledge
            if debug: print(f"mcts_env.restore_hand: Player {player} card {card} no longer valid and no other valid! Invoke DealSpecific which destroys card knowledge")
            destroy_knowledge = True
            card = self.determiniser.valid_cards(player, card_index, self.state.player_hands()
                                                 , self.state.discard_pile(), self.state.fireworks(), None
                                                 , remember_hand[card_index + 1:len(remember_hand)])

      # Now we're happy and can deal the identified card
      deal_specific_move = HanabiMove.get_deal_specific_move(card_index, player, card.color(), card.rank(), destroy_knowledge)
      if debug: print(f"mcts_env.restore_hand: Player {player} restoring card {card}")
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


  def replace_hand(self, player):
    """Redeterminise a player hand based on valid permutation
    Note: Originally this only allowed replacing of current player hand"""
    debug = False
    temp_observation = self.state.observation(player)
    card_knowledge = temp_observation.card_knowledge()[0]

    for card_index in range(len(self.state.player_hands()[player])):
      # MB: Need to remember to get card_knowledge through an assigned temp_observation
      valid_card = self.determiniser.valid_card(player, card_index, self.state.player_hands(), self.state.discard_pile()
                                                , self.state.fireworks(), card_knowledge)
      if debug: print(
        "MB: replace_hand will replace {} with {} for player {}".format(self.state.player_hands()[player][card_index],
                                                                        valid_card, player))
      assert valid_card is not None  # MB: This is where it could slip up; intra-hand conflict
      return_move = HanabiMove.get_return_move(card_index=card_index, player=player)
      self.state.apply_move(return_move)
      if debug: print("MB: replace_hand passed return move")
      # Check where this is dealt to
      deal_specific_move = HanabiMove.get_deal_specific_move(card_index, player, valid_card.color(), valid_card.rank())
      self.state.apply_move(deal_specific_move)
      if debug: print("MB: replace_hand successfully replaced that card")


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
  elif environment_name == "Hanabi-Full-Minimal":
    return MCTSEnv(
        config={
            "colors": 5,
            "ranks": 5,
            "players": num_players,
            "mcts_player":mcts_player,
            "max_information_tokens": 8,
            "max_life_tokens": 3,
            "observation_type": pyhanabi.AgentObservationType.MINIMAL.value
        })
  elif environment_name == "Hanabi-Small":
    return MCTSEnv(
        config={
            "colors":
                2,
            "ranks":
                5,
            "players":
                num_players,
            "mcts_player":
              mcts_player,
            "hand_size":
                2,
            "max_information_tokens":
                3,
            "max_life_tokens":
                1,
            "observation_type":
                pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
        })
  elif environment_name == "Hanabi-Very-Small":
    return MCTSEnv(
        config={
            "colors":
                1,
            "ranks":
                5,
            "players":
                num_players,
            "mcts_player":
                mcts_player,
            "hand_size":
                2,
            "max_information_tokens":
                3,
            "max_life_tokens":
                1,
            "observation_type":
                pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
        })
  else:
    raise ValueError("Unknown environment {}".format(environment_name))