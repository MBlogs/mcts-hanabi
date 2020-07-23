# MB Agent created during testing
from hanabi_learning_environment.rl_env import Agent
from hanabi_learning_environment.rl_env import Agent
from hanabi_learning_environment.agents.mcts.mcts_node import MCTSNode
from collections import defaultdict

class MCTSAgent(Agent):
  """Agent based on Redeterminizing Information Set Monte Carlo Tree Search"""

  def __init__(self, config, **kwargs):
    """Initialize the agent."""
    self.config = config
    self.children = dict()
    self.Q = defaultdict(int)
    self.N = defaultdict(int)

  def act(self, observation):
    """Act based on an observation. """
    if observation['current_player_offset'] != 0:
      return None
    # Determinize: Sample our cards, create perfect information state.
    # Need to then create that state for our own HanabiEnvironment
    for _ in range(100):
      self.do_rollout(self.root_node)

    def do_rollout(node):
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

    print("Computing RISMCTSAgent action...")

