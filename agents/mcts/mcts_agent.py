# MB Agent created during testing
from rl_env import Agent
from collections import defaultdict
import math
import time
from agents.rule_based.rule_based_agents import VanDenBerghAgent
from agents.mcts.mcts_node import MCTSNode
from agents.mcts import mcts_env

AGENT_CLASSES = {'VanDenBerghAgent': VanDenBerghAgent}

class MCTSAgent(Agent):
  """Agent based on Redeterminizing Information Set Monte Carlo Tree Search"""

  def __init__(self, config, **kwargs):
    """Initialize the agent."""
    # Make use of special MCTSEnv that allows redterminizing hands during rollouts
    self.environment = mcts_env.make('Hanabi-Full', num_players=config["players"], mcts_player = config['player_id'])
    self.max_information_tokens = config.get('information_tokens', 8)
    self.root_node = None
    self.root_state = None
    # MB: Nodes hashed by moves to get there
    self.exploration_weight = 2.5
    # Limits on the time or number of rollouts (whatever is first)
    self.max_time_limit = 1000 # in ms
    self.max_rollout_num = 500
    self.max_simulation_steps = 2
    # Determines the only actions to consider when branching

    # Dictionary of lists of nodes
    self.children = dict()
    self.Q = defaultdict(int)
    self.N = defaultdict(int)
    self.agents = [VanDenBerghAgent(config), VanDenBerghAgent(config), VanDenBerghAgent(config)]

  def act(self, observation, state):
    debug = False
    if observation['current_player_offset'] != 0:
      return None

    self._reset(state)

    if debug:
      print(" ################################################## ")
      print(" ################ START MCTS FORWARD MODEL ROLLOUTS ################## ")

    rollout = 0
    start_time = int(round(time.time() * 1000))
    elapsed_time = 0

    # While within rollout limit and time limit, perform rollout iterations
    while rollout < self.max_rollout_num and elapsed_time < self.max_time_limit:
      if debug: print(f" ################ START MCTS ROLLOUT: {rollout} ############## ")
      if debug: print(self.root_state)
      # Master determinisation of MCTS agent's hand
      self.environment.state = self.root_state.copy()
      self.environment.replace_hand(self.player_id)
      # Reset state of root node
      self.root_node.focused_state = self.environment.state
      if debug: print("mcts_agent.act: Player {} did master determinisation".format(self.environment.state.cur_player()))
      if debug: self.environment.print_state()
      # Rollout one iteration under this master determinisation
      reward = self._do_rollout(self.root_node, observation)
      rollout += 1
      elapsed_time = int(round(time.time() * 1000)) - start_time

      if debug:
        print(f"MB: mcts_agent.act: Tree looks like {self._get_tree_string()}")
        print(f"MB: mcts_agent.rollout_game: Game completed roll-out with reward: {reward}")
        print(f" ############### END MCTS ROLLOUT: {rollout} ################# \n")

    if debug:
      print("\n\n ################################################## ")
      print(" ################ END MCTS FORWARD MODEL ROLLOUTS ################## \n\n")

    # Now at the end of training
    print(f"mcts_agent.act: Tree looks like {self._get_tree_string()}")
    self.root_node.focused_state = self.root_state.copy()
    best_node = self._choose(self.root_node)
    print(f"mcts_agent.act: Chose node {best_node}")
    return best_node.initial_move()


  def _do_rollout(self, node, observation):
    debug = False
    # Do rollout tries to roll the focused state according to the moves in the tree

    # Select the path through tree and expansion node
    path = self._select(node)
    leaf = path[-1]
    if debug: print(f"MB: mcts_agent._do_rollout: Leaf node to roll out from is {leaf}")

    # Assign the focused_state of the node (if possible)
    for move in leaf.moves:
      if not any(move == legal_move for legal_move in self.environment.state.legal_moves()):
        if debug: print(f"MB: mcts_agent._do_rollout: move {move} not valid for this determinisation")

        # MB: If can't reach node on this determinisation, return 0 when reaching here
        # MB: Hopefully de-incetivises paths that are less likely to be the case
        reward = self.environment.state.reward()
        self._backpropagate(path, reward)
        return reward
      if debug: print(f"mcts_agent._do_rollout: Trying to step move: {move}")
      observations, reward, done, unused_info = self.environment.step(move)
      observation = observations['player_observations'][self.environment.state.cur_player()]

    leaf.focused_state = self.environment.state

    self._expand(leaf,observation)

    # Simulate from this point
    reward = self._simulate(leaf)
    self._backpropagate(path, reward)
    # Don't need to return reward but do it anyway
    return reward



  def _choose(self, node):
    ''' Choose the final move in game by best average score'''
    if node.is_terminal():
      raise RuntimeError(f"choose called on terminal node {node}")
    if node not in self.children:
      print(f"mcts_agent._choose: Choose called on node: {node}, but not in children. So finding random")
      return node.find_random_child()

    def score(n):
      if self.N[n] == 0:
        return float("-inf")  # avoid unseen moves
      return self.Q[n] / self.N[n]  # average reward

    return max(self.children[node], key=score)

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


  def _expand(self, node, observation):
    """Update the `children` dict with the children of `node
    Observation is from perspective of acting player at that node"""
    debug = False
    if node in self.children:
      if debug: print(f"mcts_agent._expand: Oops, asked to expand an already known node: {node}")
      return

    # Update children of this node. Some new moves may be promising in this determinsation
    if debug: print(f"mcts_agent._expand: Expanding children for node: {node}")
    actions = node.find_children(observation)
    moves = set([self.environment._build_move(action) for action in actions])
    self.children[node] = [MCTSNode(node.moves+(move,)) for move in moves]
    if debug: print(f"mcts_agent._expand: Took assigned node {node} and updated children {self.children[node]}")


  def _simulate(self, node):
    "MB: Returns the reward for a random simulation (to completion) of `node`"
    debug = False

    # MB: Note: The nodes state needs to be copied and determinized/sound by here
    self.environment.state = node.focused_state
    observations = self.environment._make_observation_all_players()

    done = node.is_terminal()
    reward = self.environment.reward()
    steps = 0

    while not done:
      for agent_id, agent in enumerate(self.agents):
        observation = observations['player_observations'][agent_id]
        if observation['current_player'] == agent_id:
          current_player_action = agent.act(observation)
          if debug: print(f"mcts_agent.rollout_game: Agent {agent_id} completed action {current_player_action}")
      observations, reward, done, unused_info = self.environment.step(current_player_action)
      steps += 1
      if not done:
        done = steps >= self.max_simulation_steps

    return reward


  def _backpropagate(self, path, reward):
    "Send the reward back up to the ancestors of the leaf"
    for node in reversed(path):
      self.N[node] += 1
      self.Q[node] += reward


  def _uct_select(self, node):
    "Select a child of node, balancing exploration & exploitation"
    # All children of node should already be explored (i.e appearing in master children)
    debug = False
    assert all(n in self.children for n in self.children[node])
    # Now select which leaf node of the current fully explored tree to explore nodes for
    log_N_vertex = math.log(self.N[node])
    def uct(n):
      "Upper confidence bound for trees"
      return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
        log_N_vertex / self.N[n]
      )
    selected_child = max(self.children[node], key=uct)
    if debug: print(f"mcts_agent._uct_select: Node {selected_child} was chosen as next to explore")
    return selected_child


  def _reset(self, state):
    self.player_id = state.cur_player()
    self.root_state = state.copy()
    self.root_node = MCTSNode(())
    self.children = dict()
    self.Q = defaultdict(int)
    self.N = defaultdict(int)
    self.N[self.root_node] = 0
    self.Q[self.root_node] = 0


  def _get_tree_string(self):
    tree_string = ""
    for node, children in self.children.items():
      tree_string += f"[{node}: {self.N[node]}, {self.Q[node]}] "
    return tree_string
