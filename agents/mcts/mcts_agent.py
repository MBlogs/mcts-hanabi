# MB Agent created during testing
from rl_env import Agent
from collections import defaultdict
import math
import time
from agents.mcts import mcts_env
from agents.mcts.mcts_node import MCTSNode
from agents.rule_based.ruleset import Ruleset
from agents.rule_based.rule_based_agents import VanDenBerghAgent
from agents.rule_based.rule_based_agents import FlawedAgent
from agents.rule_based.rule_based_agents import LegalRandomAgent

AGENT_CLASSES = {'VanDenBerghAgent': VanDenBerghAgent, 'FlawedAgent': FlawedAgent}

class MCTSAgent(Agent):
  """Agent based on Redeterminizing Information Set Monte Carlo Tree Search"""

  def __init__(self, config, **kwargs):
    """Initialize the agent."""
    # Dictionary of lists of nodes
    self.children = dict()
    self.Q = defaultdict(int)
    self.N = defaultdict(int)
    self.root_node = None
    self.root_state = None
    # Make use of special MCTSEnv that allows redterminizing hands during rollouts
    self.environment = mcts_env.make('Hanabi-Full', num_players=config["players"], mcts_player=config['player_id'])
    self.max_information_tokens = config.get('information_tokens', 8)
    # Limits on the time or number of rollouts (whatever is first)
    self.max_time_limit = 1000 # in ms
    self.max_rollout_num = 10
    self.max_simulation_steps = config["players"]
    self.agents = [VanDenBerghAgent(config) for _ in range(config["players"])]
    self.exploration_weight = 2.5
    # Determines the only actions to consider when branching
    self.rules = True
    self.rules = [Ruleset.tell_most_information_factory(True) # TellMostInformation
      ,Ruleset.tell_anyone_useful_card # TellAnyoneUseful
      ,Ruleset.tell_dispensable_factory(8)
      ,Ruleset.tell_playable_card_outer  # Hint missing information about a playable card
      ,Ruleset.tell_dispensable_factory(1)  # Hint full inforamtion about a disardable card
      ,Ruleset.tell_anyone_useful_card  # Hint full information about an unplayable (but not discardable) card
      ,Ruleset.play_probably_safe_factory(0.7, True)  # Play card with 70% certainty
      ,Ruleset.play_probably_safe_factory(0.4, False)  # Play card with 40% certainty and <5 cards left
      ,Ruleset.discard_probably_useless_factory(0)]

  def act(self, observation, state):
    debug = True
    if observation['current_player_offset'] != 0:
      return None

    self._reset(state)

    if debug:
      print(" ################################################## ")
      print(" ################ START MCTS FORWARD MODEL ROLLOUTS ################## ")

    rollout = 0
    start_time = time.time()
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
      self.environment.reset(observation)
      if debug: print("mcts_agent.act: Player {} did master determinisation".format(self.environment.state.cur_player()))
      if debug: self.environment.print_state()
      # Rollout one iteration under this master determinisation
      path, reward = self._do_rollout(self.root_node, observation)
      rollout += 1
      elapsed_time = (time.time() - start_time) * 1000

      if debug:
        print(f"mcts_agent.act: Path selected for roll_out was: {path}")
        print(f"mcts_agent.rollout_game: Game completed roll-out with reward: {reward}")
        print(f"mcts_agent.rollout_game: Rollout Game Stats: {self.environment.game_stats()}")
        print(f"mcts_agent.rollout_game: Rollout Player Stats: {self.environment.player_stats()}")
        print(f"mcts_agent.act: Tree updated to: {self._get_tree_string()}")
        print(f" ############### END MCTS ROLLOUT: {rollout} ################# \n")

    if debug:
      print("\n\n ################################################## ")
      print(" ################ END MCTS FORWARD MODEL ROLLOUTS ################## \n\n")

    # Now at the end of training
    if debug: print(f"mcts_agent.act: Tree looks like {self._get_tree_string()}")
    print(f"mcts_agent.act: Tree looks like {self._get_tree_string()}")
    self.root_node.focused_state = self.root_state.copy()
    best_node = self._choose(self.root_node)
    if debug: print(f"mcts_agent.act: Chose node {best_node}")
    print(f"mcts_agent.act: Chose node {best_node}")
    return best_node.initial_move()


  def _do_rollout(self, node, observation):
    # ToDO: Should not being able to get to node/tree depth backprop the full path?
    debug = True
    # Do rollout tries to roll the focused state according to the moves in the tree

    # Select the path through tree and expansion node
    path = self._select(node)
    leaf = path[-1]
    if debug: print(f"MB: mcts_agent._do_rollout: Leaf node to roll out from is {leaf}")

    # Try to get down to the selected node to roll out from it
    for move in leaf.moves:
      # If move not legal on this determinisation cut path here are backpropogate
      if (not any(move == legal_move for legal_move in self.environment.state.legal_moves())):
        reward = self.environment.reward()
        self._backpropagate(path, reward)
        return path, reward
      if debug: print(f"mcts_agent._do_rollout: Trying to step move: {move}")
      observations, reward, done, unused_info = self.environment.step(move)
      observation = observations['player_observations'][self.environment.state.cur_player()]
      if debug: print(self.environment.state)

    leaf.focused_state = self.environment.state
    self._expand(leaf, observation)
    # Simulate from this point
    reward = self._simulate(leaf)
    self._backpropagate(path, reward)
    return path, reward


  def _choose(self, node):
    ''' Choose the final move in game by best average score'''
    if node.is_terminal():
      raise RuntimeError(f"choose called on terminal node {node}")
    if node not in self.children:
      print(f"mcts_agent._choose: Choose called on node: {node}, but not in children. So finding random")
      return node.find_random_child()

    def score(n):
      if self.N[n] <= 1:
        return float("-inf")  # avoid rarely seen moves
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
    # ToDo: Update children of this node. Some new moves may be promising in this determinsation
    if debug: print(f"mcts_agent._expand: Expanding children for node: {node}")
    actions = node.find_children(observation)
    moves = set([self.environment._build_move(action) for action in actions])
    self.children[node] = [MCTSNode(node.moves+(move,), self.rules) for move in moves]
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
    self.root_node = MCTSNode((), self.rules)
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
