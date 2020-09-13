# MB Agent created during testing
from rl_env import Agent
from collections import defaultdict
import math
import time
from agents.mcts import mcts_env
from agents.mcts.mcts_node import MCTSNode
from agents.rule_based.ruleset import Ruleset
from agents.rule_based.rule_based_agents import VanDenBerghAgent
from agents.rule_based.rule_based_agents import OuterAgent
from agents.rule_based.rule_based_agents import InnerAgent
from agents.rule_based.rule_based_agents import PiersAgent
from agents.rule_based.rule_based_agents import IGGIAgent
from agents.rule_based.rule_based_agents import LegalRandomAgent
from agents.rule_based.rule_based_agents import FlawedAgent
from agents.rule_based.rule_based_agents import MuteAgent
from visualise_tree import Tree
import pyhanabi

AGENT_CLASSES = {'VanDenBerghAgent': VanDenBerghAgent,'FlawedAgent':FlawedAgent
                  , 'OuterAgent':OuterAgent, 'InnerAgent':InnerAgent, 'PiersAgent':PiersAgent, 'IGGIAgent':IGGIAgent
                  , 'LegalRandomAgent':LegalRandomAgent,'MuteAgent':MuteAgent}

class MCTSAgent(Agent):
  """Agent based on Redeterminizing Information Set Monte Carlo Tree Search"""

  def __init__(self, config):
    """Initialize the agent."""
    # Setup for tree node tracking
    self.children = dict()
    self.Q = defaultdict(int)
    self.N = defaultdict(int)
    self.root_node = None
    self.root_state = None
    self.player_id = config["player_id"]
    # Assign values based on config
    self.max_time_limit =  10000# in ms
    self.max_rollout_num = 10
    self.max_simulation_steps = config["players"]
    self.agents = [VanDenBerghAgent(config) for _ in range(config["players"])]
    self.exploration_weight = 2.5
    self.max_depth = 100
    self.determine_type = mcts_env.DetermineType.RESTORE
    self.score_type = mcts_env.ScoreType.SCORE
    self.playable_now_convention = False # Agent will follow playable now convention
    self.playable_now_convention_sim = False # Agents in simulation follow playable now convention
    self.rules =  [Ruleset.tell_most_information_factory(True)  # TellMostInformation
        , Ruleset.tell_anyone_useful_card  # TellAnyoneUseful
        , Ruleset.tell_dispensable_factory(8)
        , Ruleset.complete_tell_useful
        , Ruleset.complete_tell_dispensable
        , Ruleset.complete_tell_unplayable
        , Ruleset.play_probably_safe_factory(0.7, False)
        , Ruleset.play_probably_safe_late_factory(0.4, 5)
        , Ruleset.discard_most_confident]
    self.mcts_type = config["mcts_types"][config['player_id']]
    self._edit_mcts_config(self.mcts_type, config)
    # Make use of special MCTSEnv that allows redterminizing hands during rollouts
    self.environment = mcts_env.make('Hanabi-Full', num_players=config["players"], mcts_player=config['player_id']
                                     ,determine_type = self.determine_type, score_type = self.score_type)
    self.max_information_tokens = config.get('information_tokens', 8)
    # For Animation
    self.vis_tree = Tree()
    print(self._get_mcts_config())

  def _edit_mcts_config(self, mcts_type, config):
    """Interpret the mcts_type character"""
    if mcts_type == '0': #default
      pass
    elif mcts_type == '1': #regret
      self.score_type = mcts_env.ScoreType.REGRET
    elif mcts_type == '2': #c_regret
      self.score_type = mcts_env.ScoreType.REGRET
      self.playable_now_convention = True
      self.playable_now_convention_sim = True
    elif mcts_type == '3': #detnone
      self.determine_type = mcts_env.DetermineType.NONE
    elif mcts_type == '4': #detnone_rulesnone
      self.determine_type = mcts_env.DetermineType.NONE
      self.rules = None
    elif mcts_type == '5': #detnone_random_rulesnone
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents = [LegalRandomAgent(config) for _ in range(config["players"])]
      self.rules = None
    elif mcts_type == '6': #detnone_regret_depth1
      self.determine_type = mcts_env.DetermineType.NONE
      self.score_type = mcts_env.ScoreType.REGRET
      self.max_depth = 1
    elif mcts_type == '7': #c
      self.playable_now_convention
      self.playable_now_convention_sim
    elif mcts_type == '8': #rulesnone
      self.rules = None
    elif mcts_type == '9': #detnone_regret
      self.determine_type = mcts_env.DetermineType.NONE
      self.score_type = mcts_env.ScoreType.REGRET
    elif mcts_type == 'a': #regret_rulesnone
      self.score_type = mcts_env.ScoreType.REGRET
      self.rules = None
    elif mcts_type == 'b': #detnone_regret_rulesnone
      self.determine_type = mcts_env.DetermineType.NONE
      self.score_type = mcts_env.ScoreType.REGRET
      self.rules = None
    elif mcts_type == 'c': #detnone_c
      self.determine_type = mcts_env.DetermineType.NONE
      self.playable_now_convention = True
      self.playable_now_convention_sim = True
    elif mcts_type == 'd': #mix_default
      self.determine_type = mcts_env.DetermineType.NONE
    elif mcts_type == 'e':  # mix_flawed
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = FlawedAgent(config)
    elif mcts_type == 'f':  # mix_flawed_regret
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = FlawedAgent(config)
      self.score_type = mcts_env.ScoreType.REGRET
    elif mcts_type == 'g':  # mix_flawed_depth1
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = FlawedAgent(config)
      self.max_depth = 1
      self.max_simulation_steps = config["players"] - 1
    elif mcts_type == 'h':  # mix_flawed_regret_depth1
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = FlawedAgent(config)
      self.score_type = mcts_env.ScoreType.REGRET
      self.max_depth = 1
      self.max_simulation_steps = config["players"] - 1
    elif mcts_type == 'i':  # mix_mute
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = MuteAgent(config)
    elif mcts_type == 'j':  # mix_mute_regret
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = MuteAgent(config)
      self.score_type = mcts_env.ScoreType.REGRET
    elif mcts_type == 'k':  # mix_mute_depth1
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = MuteAgent(config)
      self.max_depth = 1
      self.max_simulation_steps = config["players"] - 1
    elif mcts_type == 'l':  # mix_mute_regret_depth1
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = MuteAgent(config)
      self.score_type = mcts_env.ScoreType.REGRET
      self.max_depth = 1
      self.max_simulation_steps = config["players"] - 1
    elif mcts_type == 'm':  # mix_inner
      self.DetermineType = mcts_env.DetermineType.NONE
      self.agents[0] = InnerAgent(config)
    elif mcts_type == 'n':  # mix_inner_regret
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = InnerAgent(config)
      self.score_type = mcts_env.ScoreType.REGRET
    elif mcts_type == 'o':  # mix_inner_depth1
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = InnerAgent(config)
      self.max_depth = 1
      self.max_simulation_steps = config["players"] - 1
    elif mcts_type == 'p':  # mix_inner_regret_depth1
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = InnerAgent(config)
      self.score_type = mcts_env.ScoreType.REGRET
      self.max_depth = 1
      self.max_simulation_steps = config["players"] - 1
    elif mcts_type == 'q':  # mix_random
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = LegalRandomAgent(config)
    elif mcts_type == 'r':  # mix_random_regret
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = LegalRandomAgent(config)
      self.score_type = mcts_env.ScoreType.REGRET
    elif mcts_type == 's':  # mix_random_depth1
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = LegalRandomAgent(config)
      self.max_depth = 1
      self.max_simulation_steps = config["players"] - 1
    elif mcts_type == 't':  # mix_random_regret_depth1
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = LegalRandomAgent(config)
      self.score_type = mcts_env.ScoreType.REGRET
      self.max_depth = 1
      self.max_simulation_steps = config["players"] - 1
    elif mcts_type == 'u':  # mix_vdb
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = VanDenBerghAgent(config)
    elif mcts_type == 'v':  # mix_vdb_regret
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = VanDenBerghAgent(config)
      self.score_type = mcts_env.ScoreType.REGRET
    elif mcts_type == 'w':  # mix_vdb_depth1
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = VanDenBerghAgent(config)
      self.max_depth = 1
      self.max_simulation_steps = config["players"] - 1
    elif mcts_type == 'x':  # mix_vdb_regret_depth1
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = VanDenBerghAgent(config)
      self.score_type = mcts_env.ScoreType.REGRET
      self.max_depth = 1
      self.max_simulation_steps = config["players"] - 1
    elif mcts_type == 'x': #fast test
      self.max_rollout_num = 10
      self.score_type = mcts_env.ScoreType.REGRET
      self.determine_type = mcts_env.DetermineType.NONE
      self.max_depth = 1
      self.agents[0] = FlawedAgent(config)
    elif mcts_type == 't': #test
      self.max_rollout_num = 25
      self.score_type = mcts_env.ScoreType.REGRET
    else:
      print(f"'mcts_config_error {mcts_type}',")

  def _get_mcts_config(self):
    return f"{{'max_time_limit':{self.max_time_limit}, 'max_rollout_num':{self.max_rollout_num}" \
           f",'agents':'{self.agents}', 'max_simulation_steps':{self.max_simulation_steps}, 'max_depth':{self.max_depth}" \
           f", 'determine_type':{self.determine_type}, 'score_type':{self.score_type}, 'exploration_weight':{self.exploration_weight}" \
           f",'playable_now_convention':{self.playable_now_convention},'playable_now_convention_sim':{self.playable_now_convention_sim}, 'rules':'{self.rules}'}}," \

  def __str__(self):
    return 'MCTSAgent'+str(self.mcts_type)

  def __repr__(self):
    return str(self)

  def act(self, observation, state):
    debug = False
    if observation['current_player_offset'] != 0:
      return None

    # Playable Now convention: If I was told a single information about a single card, and it could be playable, do it
    if self.playable_now_convention:
      action = Ruleset.playable_now_convention(observation)
      if action is not None:
        return action

    self._reset(state)

    if debug:
      print(" ################################################## ")
      print(" ################ START MCTS FORWARD MODEL ROLLOUTS ################## ")

    rollout = 0
    start_time = time.time()
    elapsed_time = 0

    # While within rollout limit and time limit, perform rollout iterations
    while rollout < self.max_rollout_num and elapsed_time < self.max_time_limit:
      if debug: print(f" ################ START {self} ROLLOUT: {rollout} ############## ")
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
      self.vis_tree.update_tree_animation(self.children, self.N, self.Q)
      elapsed_time = (time.time() - start_time) * 1000

      if debug:
        print(f"{self} mcts_agent.act: Path selected for roll_out was: {path}")
        print(f"{self} mcts_agent.rollout_game: Game completed roll-out with reward: {reward}")
        print(f"{self} mcts_agent.rollout_game: Rollout Game Stats: {self.environment.game_stats()}")
        print(f"{self} mcts_agent.rollout_game: Rollout Player Stats: {self.environment.player_stats()}")
        print(f"{self} mcts_agent.act: Tree updated to: {self._get_tree_string()}")
        print(f" ############### END MCTS ROLLOUT: {rollout} ################# \n")

    if debug:
      print("\n\n ################################################## ")
      print(" ################ END MCTS FORWARD MODEL ROLLOUTS ################## \n\n")

    # Now at the end of training
    if debug: print(f"mcts_agent.act: Tree looks like {self._get_tree_string()}")

    self.root_node.focused_state = self.root_state.copy()
    best_node = self._choose(self.root_node)
    if debug: print(f"mcts_agent.act: Chose node {best_node}")
    #print(f"mcts_agent.act: Tree looks like {self._get_tree_string()}")
    #print(f"mcts_agent.act: Chose node {best_node}")
    self.vis_tree.create_tree_animation()
    return best_node.initial_move()


  def _do_rollout(self, node, observation):
    # ToDO: Should not being able to get to node/tree depth backprop the full path?
    debug = False
    # Do rollout tries to roll the focused state according to the moves in the tree

    # Select the path through tree and expansion node
    path = self._select(node)
    leaf = path[-1]
    if debug: print(f"MB: mcts_agent._do_rollout: Leaf node to roll out from is {leaf}")

    # Try to get down to the selected node to roll out from it
    depth = 0
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
      depth += 1
      if depth > self.max_depth:
        break
      #ToDO: This seems debateable...

    leaf.focused_state = self.environment.state
    # Don't expand if we didn't get to a root
    if not depth > self.max_depth:
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
    if debug: print(f"mcts_agent._expand: Expanding children for node: {node}")

    moves = node.find_children(observation)
    # Need it in move form. If in action form, convert them
    if len(moves) > 0 and isinstance(moves[0], dict):
      moves = set([self.environment._build_move(action) for action in moves])
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

          # Playable now convention
          if self.playable_now_convention_sim:
            playable_now_action = Ruleset.playable_now_convention(observation)
            if playable_now_action is not None:
              current_player_action == playable_now_action

      observations, reward, done, unused_info = self.environment.step(current_player_action)
      if debug: print(f"mcts_agent.rollout_game: Agent {agent_id} completed action {current_player_action}")
      steps += 1
      #print(f"mcts_agent.simulate steps are {steps}")
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
