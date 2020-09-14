import matplotlib
matplotlib.use("TKAgg")
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
    self.trees.append(tree.copy())
    self.Ns.append(N.copy())
    self.Qs.append(Q.copy())

  def create_tree_animation(self):
    plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
    fig, ax = plt.subplots(figsize=(14, 12))

    def tree_animation_frame(i):
      ax.clear()
      develop_tree(self.Ns[i], self.Qs[i])
      ax.plot()

    def develop_tree(N, Q):
      """Pass in list of node dicts with path as key"""
      size_min = 100
      size_max = 400
      node_labels = {}
      node_labels['root'] = round(max(Q.values()) / max(N.values()), 1)
      edge_labels = {}
      node_N = [max(N.values())]
      node_Q = [max(Q.values())]
      edges = []
      for node, ns in N.items():
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
      # If all values are the same, set node_color all to 1 to avoid div by 0
      min_node_avg = min(node_avg)
      max_node_avg = max(node_avg)

      if max_node_avg - min_node_avg == 0:
        node_color = [0 for val in node_avg]
      else:
        node_color = [(val - min_node_avg) / (max_node_avg - min_node_avg) for val in node_avg]
      node_size = [n * (size_max - size_min) + size_min for n in node_avg]

      # Cycled through all the nodes so print
      G = nx.Graph()
      #print(edges)
      G.add_edges_from(edges)
      pos = graphviz_layout(G, prog='dot')
      nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_color=node_color, cmap=plt.cm.Blues_r)
      nx.draw_networkx_edges(G, pos)
      nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax, font_size=8)
      nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, label_pos=0.5, font_size=6)

    # Make and save animation
    ani = animation.FuncAnimation(fig, tree_animation_frame,frames=len(self.trees), interval=20,repeat_delay=5000)
    # Finally, show live animation
    plt.show()
    waiting = input()

  def _get_tree_string(self, tree, N, Q):
    tree_string = ""
    for node, children in tree.items():
      tree_string += f"[{node}: {N[node]}, {Q[node]}] "
    return tree_string