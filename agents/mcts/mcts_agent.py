# MB Agent created during testing
import rl_env
from rl_env import Agent
from collections import defaultdict
import math
from agents.rule_based_agent import RuleBasedAgent
from agents.mcts.mcts_node import MCTSNode

AGENT_CLASSES = {'RuleBasedAgent':RuleBasedAgent}

class MCTSAgent(Agent):
  """Agent based on Redeterminizing Information Set Monte Carlo Tree Search"""

  def __init__(self, config, **kwargs):
    """Initialize the agent."""
    # ToDo: Needs to know all HanabiEnv parameters
    self.environment = rl_env.make('Hanabi-Full', num_players=config["players"])
    self.max_information_tokens = config.get('information_tokens', 8)
    self.root_node = None
    self.root_state = None
    # MB: Nodes hashed by moves to get there
    self.exploration_weight = 0.1
    self.rollout_num = 100
    self.max_simulation_steps = 2
    # Dictionary of lists of nodes
    self.children = dict()
    self.Q = defaultdict(int)
    self.N = defaultdict(int)
    self.agents = [RuleBasedAgent(config), RuleBasedAgent(config), RuleBasedAgent(config)]

  def act(self, observation, state):
    debug = True
    if observation['current_player_offset'] != 0:
      return None

    self._reset(state)

    if debug:
      print(" ################################################## ")
      print(" ################ START MCTS FORWARD MODEL ROLLOUTS ################## ")

    for r in range(self.rollout_num):
      if debug: print(f" ################ START MCTS ROLLOUT: {r} ############## ")
      # Reset state of root node and environment
      self.root_node.state = self.root_state.copy()
      # Master determinisation
      self.root_node.state.replace_hand()
      self.environment.state = self.root_node.state
      print("MB: Player {} replaced hand".format(self.environment.state.cur_player()))
      reward = self._do_rollout(self.root_node)
      if debug:
        print(f"MB: mcts_agent.act: Tree looks like {self._get_tree_string()}")
        print(f"MB: mcts_agent.rollout_game: Game completed roll-out with reward: {reward}")
        print(f" ############### END MCTS ROLLOUT: {r} ################# \n")

    if debug:
      print("\n\n ################################################## ")
      print(" ################ END MCTS FORWARD MODEL ROLLOUTS ################## \n\n")

    # Now at the end of training, so choose best
    best_node = self._choose(self.root_node)
    return best_node.initial_move()


  def _do_rollout(self, node):
    debug = True

    # Select the path through tree and expansion node
    path = self._select(node)
    leaf = path[-1]

    # Assign the focused_state of the node (if possible)
    for move in leaf.moves:
      if not any(move == legal_move for legal_move in self.environment.state.legal_moves()):
        if debug: print(f"MB: mcts_agent._do_rollout: move {move} not valid for this determinisation")
        return -1
      observations, reward, done, unused_info = self.environment.step(move)

    leaf.focused_state = self.environment.state
    self._expand(leaf)
    reward = self._simulate(leaf)
    self._backpropagate(path, reward)
    # Don't need to return reward but do it anyway
    return reward


  def _choose(self, node):
    ''' Choose move in game '''
    # ToDO: How to handle terminal nodes?
    if node.is_terminal():
      raise RuntimeError(f"choose called on terminal node {node}")
    if node not in self.children:
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


  def _expand(self, node):
    "Update the `children` dict with the children of `node`"
    if node in self.children:
      return  # already expanded
    self.children[node] = node.find_children()


  def _simulate(self, node):
    "MB: Returns the reward for a random simulation (to completion) of `node`"
    debug = True

    # MB: Note: The nodes state needs to be copied and determinized/sound by here
    self.environment.state = node.focused_state
    observations = self.environment._make_observation_all_players()

    done = False
    steps = 0
    while not done:
      for agent_id, agent in enumerate(self.agents):
        observation = observations['player_observations'][agent_id]
        if observation['current_player'] == agent_id:
          current_player_action = agent.act(observation)
          if debug: print(f"MB: mcts_agent.rollout_game: Agent {agent_id} completed action {current_player_action}")

      observations, reward, done, unused_info = self.environment.step(current_player_action)
      steps += 1
      if not done: done = steps > self.max_simulation_steps
    return reward / 25.0


  def _backpropagate(self, path, reward):
    "Send the reward back up to the ancestors of the leaf"
    for node in reversed(path):
      self.N[node] += 1
      self.Q[node] += reward


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

  def _reset(self, state):
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


  def old_act(self, observation, state):
    """MB: Act based on an observation. """
    debug = True
    if observation['current_player_offset'] != 0:
      return None
    if debug: print("MB: mcts_agent.act: Deciding MCTS action")

    self.root_state = state
    self.rollout_game()

    if debug: print(" ################################################## ")
    if debug: print(" ############### END MCTS ROLLOUT ################# \n\n")
    # Determinize: Sample our cards, create perfect information state.
    # Need to then create that state for our own HanabiEnvironment
    return self.agents[0].act(observation)


  def rollout_game(self):
    debug = True
    # MB: Hack, access the protected method
    if debug:
      print("\n\n ##################################################  ")
      print(" ################ START MCTS ROLLOUT ############## ")
    # MB: Test not copying
    self.environment.state = self.root_state.copy()
    # if debug: print(f"MB: mcts_agent.rollout_game: Copied state fireworks is: {self.environment.state.fireworks()}")
    # The observations are the thing that messes it up
    observations = self.environment._make_observation_all_players()
    # if debug: print(f"MB: Copied state. The new state after observation {self.environment.state}")
    # if debug: print(f"MB: mcts_agent.rollout_game: At start example observation: \n{observations['player_observations'][0]}\n")
    # if debug: print(f"MB: mcts_agent.rollout_game: fireworks are: {self.environment.state.fireworks()}")
    done = False
    while not done:
      for agent_id, agent in enumerate(self.agents):
        observation = observations['player_observations'][agent_id]
        action = agent.act(observation)
        if observation['current_player'] == agent_id:
          assert action is not None
          if debug: print(f"MB: mcts_agent.rollout_game: Got agent {agent_id} action {action}")
          current_player_action = action
        else:
          assert action is None
      observations, reward, done, unused_info = self.environment.step(current_player_action)
      # MB: for some reason, state has all 5 fireworks but observation only has 3.
    print(f"MB: mcts_agent.rollout_game: Game completed roll-out with reward: {reward}")


'''
1. How to reach a node state in _select when the actions to get there might not be compatible with new master determinisation?
2. 
'''
