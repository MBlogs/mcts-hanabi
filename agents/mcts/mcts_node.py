import random
from abc import ABC, abstractmethod


class Node(ABC):
    """A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """
    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True


class MCTSNode(Node):
  def __init__(self, moves,rules):
    # MB: moves is a tuple of moves that uniquely identify this node
    self.moves = moves
    # MB: state is subject to change based on determinisation
    self.focused_state = None
    # Rules deciding how to expand from this node
    self.rules = rules

  def find_children(self, observation):
    "All possible successors of this board state"
    # MB: States are determined by the moves to get there
    # MB: So this is technically only one version of possible children
    # MB: Node needs a focused state to get next possible moves from
    debug = False

    assert self.focused_state is not None
    if self.is_terminal():
      if debug: print("MB: mcts_node.find_cildren: was called on terminal state. Returning empty")
      return []

    # Rulesets returns in action dict form. Return these for mcts_agent to build into moves
    if self.rules is not None:
      actions_by_rules = [rule(observation) for rule in self.rules]
      if debug: print(f"mcts_node.find_children: Found actions: {actions_by_rules}")
      children = [action for action in actions_by_rules if action is not None]
    else:
      children = self.focused_state.legal_moves()

    # Note: Could return duplicates
    return children

  def find_random_child(self):
    "Random successor of this board state (for more efficient simulation)"
    return random.choice(self.find_children())

  def is_terminal(self):
    "Returns True if the node has no children"
    return self.focused_state.is_terminal()

  def initial_move(self):
    """MB: Returns first move to get to node (if this node is best)"""
    return self.moves[0]

  def __str__(self):
    return f"{self.moves}"

  def __repr__(self):
    return f"{self.moves}"

  def __hash__(self):
    "Nodes must be hashable"
    return hash(self.moves)

  def __eq__(node1, node2):
    "Nodes must be comparable"
    # MB: Two nodes are equivalent if moves to get there are the same
    return node1.moves == node2.moves
