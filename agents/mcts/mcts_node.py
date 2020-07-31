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
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True


class MCTSNode(Node):
  def __init__(self, moves, focused_state):
    # MB: moves_list distinctly defines this node
    # MB: state is subject to change based on determinisation
    self.focused_state = focused_state
    # MB: moves is a tuple of moves that uniquely identify this node
    self.moves = moves
    self.visits = 0
    self.rewards = 0


  def find_children(self):
    "All possible successors of this board state"
    # MB: States are determined by the moves to get there
    # MB: So this is technically only one version of possible children
    # Node needs a focused state to get next possible moves from
    assert self.focused_state is not None
    children = [MCTSNode(self.moves + (move,), None) for move in self.focused_state.legal_moves()]
    return children

  def find_random_child(self):
    "Random successor of this board state (for more efficient simulation)"
    return random.choice(self.state.legal_moves())


  def is_terminal(self):
    "Returns True if the node has no children"
    return self.focused_state.is_terminal()

  def initial_move(self):
    """MB: Returns first move to get to node (if this node is best)"""
    return self.moves[0]


  def reward(self):
    "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
    return self.focused_state.fireworks_score()


  def __hash__(self):
    "Nodes must be hashable"
    return hash(self.moves)


  def __eq__(node1, node2):
    "Nodes must be comparable"
    # MB: Two nodes are equivalent if moves to get there are the same
    return node1.moves == node2.moves
