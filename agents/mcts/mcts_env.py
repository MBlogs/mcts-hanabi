import pyhanabi
from rl_env import HanabiEnv
from agents.mcts.mcts_determinizer import MCTSDeterminizer
from pyhanabi import HanabiMove


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

    # If acting player not me, restore as best as possible their hand
    #if action_player != self.mcts_player:
    #  self.state.restore_hand(action_player, self.remember_hand, actioned_card, action.card_index())
    #  if debug: print(f"mcts_env.step: Player {action_player} restored hand")

    # If cur_player is now chance, action player still needs a random card dealt
    while self.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
      self.state.deal_random_card()
      if debug: print(f"mcts_env.step: Player {self.state.cur_player()} dealt random card")

    # Now for sure onto next player. If not me, remember then replace their hand
    if self.state.cur_player() != self.mcts_player:
      self.remember_hand = self.state.player_hands()[self.state.cur_player()]
      self.replace_hand(self.state.cur_player())
      if debug: print(f"mcts_env.step: Player {self.state.cur_player()} replaced hand")

    # Now make observation, as the hand is now re-determinised
    observation = self._make_observation_all_players()
    if debug: self.print_state()

    reward = self.state.reward()
    done = self.state.is_terminal()
    info = {}
    return (observation, reward, done, info)


  def restore_hand(self, player, remember_hand, removed_card=None, removed_card_index=-1):
    """As best as possible, restore current player hand to the one passed in
    remember_hand: Their hand before it was replaced on their turn
    removed_card_index: This card was played or discarded. Used to skip over it.
    """
    debug = True

    # Work out card knowledge upfront
    temp_observation = self.state.observation(player)
    card_knowledge = temp_observation.card_knowledge()[0]

    # ToDo: Need to adjust ordering here so vaid_cards can be called successfully

    # Start by returning all cards to deck (resolves intra-hand conflict)
    for card_index in range(len(self.state.player_hands()[player])):
      # Return the card in their current position (return will always be the oldest card
      return_move = HanabiMove.get_return_move(card_index=0, player=player)
      if debug: print(f"pyhanabi.restore_hand: Player {player} returning card index: {self.state.player_hands()[player][0]}")
      self.state.apply_move(return_move)
      if debug: print(f"pyhanabi.restore_hand: Player {player} hand now {self.state.player_hands()[player]}")

    # Then deal back all cards
    card_index = 0

    for remember_card_index in range(len(remember_hand)):
      # card_index is current hand iterator (maxes at 3 OR 4)
      # remember_card_index is remember hand iterator (always maxes at 4)

      # Assign card in remembered hand to return
      card = remember_hand[remember_card_index]

      # Skip over the remembered card if it was played
      if remember_card_index == removed_card_index:
        if debug: print(f"pyhanabi.restore_hand: Player {player} skipped over restoring card {card}")
        continue


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