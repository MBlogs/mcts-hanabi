# MB Agent created during testing
from rl_env import Agent
from collections import defaultdict
import math
import time
from agents.rule_based.ruleset import Ruleset
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
    self.max_time_limit = 200 # in ms
    self.max_rollout_num = 30
    self.max_simulation_steps = 2
    # Determines the only actions to consider when branching
    self.expansion_rules = [Ruleset.tell_most_information # Hint the most information
                            , Ruleset.tell_playable_card # Hint about a playable card
                            , Ruleset.tell_anyone_useless_card # Hint about a card that can be discarded
                            , Ruleset.tell_playable_card_outer # Hint missing information about a playable card
                            , Ruleset.tell_dispensable_factory # ?Hint full information about a discardable card
                            , Ruleset.tell_anyone_useful_card # Hint full information about an unplayable (but not discardable) card
                            , Ruleset.play_if_certain #
                            , Ruleset.play_probably_safe_factory # Play a card if 70% playable
                            , Ruleset.play_probably_safe_factory # Play a card if 40% playable and 5 or fewer cards
                            , Ruleset.discard_probably_useless_factory # Discard most confident
    ]
    # Dictionary of lists of nodes
    self.children = dict()
    self.Q = defaultdict(int)
    self.N = defaultdict(int)
    self.agents = [VanDenBerghAgent(config), VanDenBerghAgent(config), VanDenBerghAgent(config)]

  def act(self, observation, state):
    debug = True
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
      if debug: print(f" ################ START MCTS ROLLOUT: {r} ############## ")
      if debug: print(self.root_state)
      # Master determinisation of MCTS agent's hand
      self.environment.state = self.root_state.copy()
      self.environment.replace_hand(self.player_id)
      # Reset state of root node
      self.root_node.focused_state = self.environment.state
      if debug: print("mcts_agent.act: Player {} did master determinisation".format(self.environment.state.cur_player()))
      # Rollout one iteration under this master determinisation
      reward = self._do_rollout(self.root_node)
      rollout += 1
      elapsed_time = int(round(time.time() * 1000)) - start_time

      if debug:
        print(f"MB: mcts_agent.act: Tree looks like {self._get_tree_string()}")
        print(f"MB: mcts_agent.rollout_game: Game completed roll-out with reward: {reward}")
        print(f" ############### END MCTS ROLLOUT: {r} ################# \n")

    if debug:
      print("\n\n ################################################## ")
      print(" ################ END MCTS FORWARD MODEL ROLLOUTS ################## \n\n")

    # Now at the end of training
    print(f"mcts_agent.act: Tree looks like {self._get_tree_string()}")
    self.root_node.focused_state = self.root_state.copy()
    best_node = self._choose(self.root_node)
    print(f"mcts_agent.act: Chose node {best_node}")
    return best_node.initial_move()


  def _do_rollout(self, node):
    debug = True
    # Do rollout tries to roll the focused state according to the moves in the tree

    # Select the path through tree and expansion node
    path = self._select(node)
    leaf = path[-1]
    if debug: print(f"MB: mcts_agent._do_rollout: Leaf node to roll out from is {leaf}")

    # Assign the focused_state of the node (if possible)
    steps = 0
    for move in leaf.moves:
      if not any(move == legal_move for legal_move in self.environment.state.legal_moves()):
        if debug: print(f"MB: mcts_agent._do_rollout: move {move} not valid for this determinisation")

        # MB: If can't reach node on this determinisation, return 0 when reaching here
        # MB: Hopefully de-incetivises paths that are less likely to be the case
        reward = 0
        self._backpropagate(path, reward)
        return reward
      if debug: print(f"mcts_agent._do_rollout: Trying to step move: {move}")
      observations, reward, done, unused_info = self.environment.step(move)
      steps += 1

    leaf.focused_state = self.environment.state

    self._expand(leaf)

    # Simulate from this point
    reward = self._simulate(leaf, steps)
    self._backpropagate(path, reward)
    # Don't need to return reward but do it anyway
    return reward



  def _choose(self, node):
    ''' Choose move in game '''
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

# 1Hintthemostinformation(not previouslygiven) toanyotherplayer
# 2Hintaboutaplayablecard
# 3Hintaboutacardthatcanbediscarded
# 4Hintmissinginformationaboutaplayablecard
# 5Hintfullinformationaboutadiscardablecard
# 6Hintfullinformationaboutanunplayable(butnot discardable) card
# 7Playacard if weareatleast70 % confidentit is playable
# 8Playacard is weareatleast40 % confidentit is playable, and wehave5 or fewercardsleft in thedeck
# 9Discardthecardthattheplayer is mostconfident is discardable

  def _expand(self, node):
    "Update the `children` dict with the children of `node`"
    if node in self.children:
      return  # already expanded
    self.children[node] = node.find_children()


  def _simulate(self, node, steps):
    "MB: Returns the reward for a random simulation (to completion) of `node`"
    debug = False

    # MB: Note: The nodes state needs to be copied and determinized/sound by here
    self.environment.state = node.focused_state
    observations = self.environment._make_observation_all_players()

    done = node.is_terminal()
    reward = self.environment.reward()
    while not done:
      for agent_id, agent in enumerate(self.agents):
        observation = observations['player_observations'][agent_id]
        if observation['current_player'] == agent_id:
          current_player_action = agent.act(observation)
          if debug: print(f"MB: mcts_agent.rollout_game: Agent {agent_id} completed action {current_player_action}")
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
