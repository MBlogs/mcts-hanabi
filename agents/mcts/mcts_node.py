class MCTSNode:
    def __init__(self, state):
        self.state = state
        self.visits = 0
        self.rewards = 0

    def find_children(self):
        "All possible successors of this board state"
        return set()

    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True
