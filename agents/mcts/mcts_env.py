import pyhanabi
from rl_env import HanabiEnv



class MCTSEnv(HanabiEnv):
  def __init__(self, config):
    # This is the forward model for a particular MCTS player. Note it's position
    self.mcts_player = config['mcts_player']
    self.remember_hand = None
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

    # Apply the action
    if debug: print(f"MB: mcts_env.step: Player {self.state.cur_player()} applying action {action}")
    action_player = self.state.cur_player()
    self.state.apply_move(action)

    # If acting player not me, restore as best as possible their hand, passing if a card was removed
    if action_player != self.mcts_player:
      if action.type() == pyhanabi.HanabiMoveType.DISCARD or action.type() == pyhanabi.HanabiMoveType.PLAY:
        self.state.restore_hand(action_player, self.remember_hand, action.card_index())
      else:
        self.state.restore_hand(action_player, self.remember_hand)
      if debug: print(f"mcts_env.step: Player {action_player} restored hand")

    # If cur_player is now chance, action player still needs a random card dealt
    while self.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
      self.state.deal_random_card()
      if debug: print(f"mcts_env.step: Player {self.state.cur_player()} dealt random card")

    # Now for sure onto next player. If not me, remember then replace their hand
    if self.state.cur_player() != self.mcts_player:
      self.remember_hand = self.state.player_hands()[self.state.cur_player()]
      self.state.replace_hand()
      if debug: print(f"mcts_env.step: Player {self.state.cur_player()} replaced hand")

    # Now make observation, as the hand is now re-determinised
    observation = self._make_observation_all_players()
    if debug: self.print_state()

    reward = self.state.reward()
    done = self.state.is_terminal()
    info = {}
    return (observation, reward, done, info)


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