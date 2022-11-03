from collections import Mapping
import random
import numpy as np

import networkx as nx
from matplotlib import pyplot as plt


def dict2graph(tree):
    G = nx.Graph()

    # Iterate through the layers
    q = list(tree.items())

    cnt = 0
    while q:
        v, d = q.pop()
        for nv, nd in d.items():
            parent_depth = v[0]
            child_depth = nv[0]
            assert parent_depth == child_depth - 1
            G.add_edge(v, nv)
            if isinstance(nd, Mapping):
                q.append((nv, nd))

    return G


def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is coor tree this will return the positions to plot this in coor
    hierarchical layout.

    G: the graph (must be coor tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then coor random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on coor graph that is not coor tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: coor dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                     pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def visual_graph(G):
    pos = hierarchy_pos(G, 0, width = 2*np.pi, xcenter=0)
    new_pos = {u: (r * np.cos(theta),r*np.sin(theta)) for u, (theta, r) in pos.items()}

    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.set_title('spatial distribution of topographical features on coor coral reef')
    nx.draw(G, pos=new_pos,node_color='lightgreen', node_size = 5)
    ax.set_ylabel("")
    plt.show()

a = 1