# MB Agent created during testing
import rl_env
from rl_env import Agent
from collections import defaultdict
import math
from agents.rule_based_agent import RuleBasedAgent
AGENT_CLASSES = {'RuleBasedAgent':RuleBasedAgent}

class MCTSAgent(Agent):
  """Agent based on Redeterminizing Information Set Monte Carlo Tree Search"""

  def __init__(self, config, **kwargs):
    """Initialize the agent."""
    # ToDo: Needs to know all HanabiEnv parameters
    self.environment = rl_env.make('Hanabi-Full', num_players=config["players"])
    self.children = dict()
    self.Q = defaultdict(int)
    self.N = defaultdict(int)
    self.root_state = None
    self.agents = [RuleBasedAgent(config), RuleBasedAgent(config), RuleBasedAgent(config)]

  def rollout_game(self, root_state):
    debug = True
    # MB: Hack, access the protected method
    if debug: print(f"MB: mcts_agent.rollout_game: fireworks are: {self.environment.state.fireworks()}")
    self.environment.state = root_state.copy(self.environment.game)
    observations = self.environment._make_observation_all_players()
    if debug: print(f"MB: mcts_agent.rollout_game: At start example observation: \n{observations['player_observations'][0]}\n")
    if debug: print(f"MB: mcts_agent.rollout_game: fireworks are: {self.environment.state.fireworks()}")
    done = False
    while not done:
      for agent_id, agent in enumerate(self.agents):
        observation = observations['player_observations'][agent_id]
        action = agent.act(observation)
        if debug: print("MB: mcts_agent.rollout_game: Got an agent action")
        if observation['current_player'] == agent_id:
          assert action is not None
          current_player_action = action
        else:
          assert action is None
      if debug: print(f"MB: mcts_agent.rollout_game: About to make the step")
      observations, reward, done, unused_info = self.environment.step(current_player_action)
      if debug: print(f"MB: mcts_agent.rollout_game: Made a step. Fireworks score currently: {reward}")
      # MB: for some reason, state has all 5 fireworks but observation only has 3.
      if debug: print(f"MB: mcts_agent.rollout_game: Example observation: \n{observations['player_observations'][0]}\n")
      if debug: print(f"MB: mcts_agent.rollout_game: State after step: \n{self.environment.state}\n")
    print(f"MB: mcts_agent.rollout_game: Game completed roll-out with reward: {reward}")

  def act(self, observation, state):
    """MB: Act based on an observation. """
    debug = True
    if observation['current_player_offset'] != 0:
      return None
    if debug: print("MB: mcts_agent.act: Deciding MCTS action")
    self.root_state = state
    # MB: Need to remember to randomise MCTSAgent hand before rolling out anything
    self.rollout_game(self.root_state)
    if debug: print("MB: mcts_agent.act: A game fully completed roll out ")
    # Determinize: Sample our cards, create perfect information state.
    # Need to then create that state for our own HanabiEnvironment

    return self.agents[0].act(observation)

  def do_rollout(self, node):
    path = self._select(node)
    leaf = path[-1]
    self._expand(leaf)
    reward = self._simulate(leaf)
    self._backpropagate(path, reward)

  def _select(self, node):
    "Find an unexplored descendent of `node`"
    path = []
    while True:
      path.append(node)
      if node not in self.children or not self.children[node]:
        # node is either unexplored or terminal
        return path
      unexplored = self.children[node] - self.children.keys()
      if unexplored:
        n = unexplored.pop()
        path.append(n)
        return path
      node = self._uct_select(node)  # descend a layer deeper

  def _expand(self, node):
    "Update the `children` dict with the children of `node`"
    if node in self.children:
      return  # already expanded
    self.children[node] = node.find_children()

  def _simulate(self, node):
    "Returns the reward for a random simulation (to completion) of `node`"
    while True:
      if node.is_terminal():
        reward = node.reward()
        return reward
      node = node.find_random_child()

  def _backpropagate(self, path, reward):
    "Send the reward back up to the ancestors of the leaf"
    for node in reversed(path):
      self.N[node] += 1
      self.Q[node] += reward
      reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

  def _uct_select(self, node):
    "Select a child of node, balancing exploration & exploitation"
    # All children of node should already be expanded:
    assert all(n in self.children for n in self.children[node])
    log_N_vertex = math.log(self.N[node])
    def uct(n):
      "Upper confidence bound for trees"
      return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
        log_N_vertex / self.N[n]
      )
    return max(self.children[node], key=uct)

