import matplotlib.pyplot as plt
from matplotlib import animation
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import random

def move2str(move):
  return str(move).replace('player ', '')

class Tree(object):
  def __init__(self):
    self.trees = []
    self.Ns = []
    self.Qs = []

  def update_tree_animation(self,tree,N,Q):
    self.trees.append(tree)
    self.Ns.append(N)
    self.Qs.append(Q)

  def create_tree_animation(self):
    fig,ax = plt.subplots(figsize=(8,12))
    print(self.Ns)

    def tree_animation_frame(i):
      ax.clear()
      develop_tree(self.trees[i], self.Ns[i], self.Qs[i])
      ax.plot()

    def develop_tree(tree, N, Q):
      """Pass in list of node dicts with path as key"""
      G = nx.Graph()
      size_min = 100
      size_max = 500
      node_labels = {"root": round(max(Q.values()) / max(N.values()), 1)}
      edge_labels = {}
      node_N = [max(N.values())]
      node_Q = [max(Q.values())]
      edges = []
      print(f"Tree is: {self._get_tree_string(tree, N, Q)}")
      for node, children in tree.items():
        # Take care around the root
        if len(node.moves) == 0:
          continue
        elif len(node.moves) == 1:
          parent = "root"
          child = node.moves
        else:
          # Else, we're interested in the end two of the path
          parent = node.moves[:-1]
          child = node.moves
        edges.append((move2str(parent), move2str(child)))
        # Add to label mapping dict, only the last move in the path
        node_labels[move2str(child)] = round(Q[node] / N[node], 2)
        edge_labels[(move2str(parent), move2str(child))] = move2str(child[-1])
        node_N.append(N[node])
        node_Q.append(Q[node])
      # Perform math on the summary stats
      node_avg = [q / n for q, n in zip(node_Q, node_N)]
      node_color = [(max(node_avg) - val) / (max(node_avg) - min(node_avg)) for val in node_avg]
      node_size = [n * (size_max - size_min) + size_min for n in node_avg]
      # Cycled through all the nodes so print
      G.add_edges_from(edges)
      pos = graphviz_layout(G, prog='dot')
      nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_color=node_color, cmap=plt.cm.Blues)
      nx.draw_networkx_edges(G, pos)
      nx.draw_networkx_labels(G, pos, node_labels, ax=ax, font_size=8)
      nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, label_pos=0.5, font_size=8)

    ani = animation.FuncAnimation(fig, tree_animation_frame,frames=len(self.trees), interval=1000,repeat_delay=1000)
    ax.clear()
    plt.show()


  def _get_tree_string(self, tree, N, Q):
    tree_string = ""
    for node, children in tree.items():
      tree_string += f"[{node}: {N[node]}, {Q[node]}] "
    return tree_string