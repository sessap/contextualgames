import pandas
import networkx as nx
import itertools
import numpy as np
import matplotlib.pyplot as plt
import os


def k_shortest_paths(G , source, target, k, weight=None):
    return list(itertools.islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

class NetworkData:
     def __init__(self, Nodes, Edges, Capacities, Freeflowtimes, Powers):
        self.Nodes   = np.reshape(Nodes, (-1,1) )
        self.Edges   = np.reshape(Edges, (-1,2) ) 
        self.Capacities   = np.reshape(Capacities, (-1,1))
        self.Freeflowtimes     = np.reshape(Freeflowtimes   , (-1,1))
        self.Powers        = np.reshape(Powers, (-1,1))

## Create Sioux-Falls Network
def Create_Network():
    SiouxNetwork = nx.DiGraph()

    reader = pandas.read_csv("SiouxFallsNet/SiouxFalls_node.csv")
    x_coords = reader['X'].values
    y_coords = reader['Y'].values
    for num in range(24):
        SiouxNetwork.add_node(str(num+1) , pos = (x_coords[num], y_coords[num]) )

    reader = pandas.read_csv("SiouxFallsNet/SiouxFalls_net.csv")
    init_nodes  = reader['Init_node'].values
    term_nodes  = reader['Term_node'].values
    lengths     = reader['Length'].values

    for e in range(len(init_nodes)):
        SiouxNetwork.add_edge(str(init_nodes[e]), str(term_nodes[e]) , weight = lengths[e])

    Nodes = np.arange(1, 25)
    Edges = np.array([init_nodes, term_nodes]).T
    Capacities = reader['Capacity'].values / 100
    Freeflowtimes = reader['Free_Flow_Time'].values
    Powers = reader['Power'].values
    SiouxNetwork_data = NetworkData(Nodes, Edges, Capacities, Freeflowtimes, Powers)

    return   SiouxNetwork, SiouxNetwork_data

def get_edge_idx (Edges, node1, node2):
    idx = np.where(np.all(Edges == [int(node1), int(node2)] ,axis=1))
    return idx

def Compute_Strategy_vectors(OD_demands, Freeflowtimes, Networkx, Edges, num_routes = 5, mult_factor = None):
    if mult_factor is None:
        multipl_factor = 3
    else:
        multipl_factor = mult_factor
    OD_pairs = []
    Demands = []
    for i in range(24):
        for j in range(24):
            if OD_demands[i, j] > 0:
                OD_pairs.append([i + 1, j + 1])
                Demands.append(OD_demands[i, j] / 100)

    E = len(Edges)
    K = num_routes  # K shortest paths for each agent
    Strategy_vectors = [()]*len(OD_pairs)
    for i in range(len(OD_pairs)):
        Strategy_vectors[i] = list()
        OD_pair = np.array(OD_pairs[i])
        paths = k_shortest_paths(Networkx, str(OD_pair[0]), str(OD_pair[1]), K, weight = 'weight')
        for a in range(len(paths)):
            vec = np.zeros((E,1))
            for n in range(len(paths[a])-1):
                idx = get_edge_idx(Edges,  paths[a][n], paths[a][n+1])
                vec[idx] = 1
            strategy_vec = np.multiply(vec, Demands[i])
            if a == 0:
                Strategy_vectors[i].append(  strategy_vec )
            if a > 0 and np.dot(strategy_vec.T, Freeflowtimes) < multipl_factor* np.dot(Strategy_vectors[i][0].T, Freeflowtimes ):
                Strategy_vectors[i].append(strategy_vec )

    return Strategy_vectors, OD_pairs


def Compute_traveltimes(NetworkData, Strategy_vectors, played_actions, player_id , Capacities = None):
    N = len(Strategy_vectors) # number of players
    Total_occupancies = np.sum([Strategy_vectors[i][played_actions[i]] for i in range(N)], axis = 0)
    if Capacities is None:
        Capacities = NetworkData.Capacities
    E = np.size(NetworkData.Edges,0)
    a = NetworkData.Freeflowtimes
    b = np.divide( np.multiply(NetworkData.Freeflowtimes, 0.15*np.ones((E,1))) , np.power(Capacities, NetworkData.Powers))
    unit_times = a + np.multiply(b, np.power(Total_occupancies, NetworkData.Powers) )
    if player_id == 'all':
        Traveltimes = np.zeros(N)
        for i in range(N):
            X_i = np.array(Strategy_vectors[i][played_actions[i]])
            Traveltimes[i] = np.dot(X_i.T, unit_times )
    else:
        X_i = Strategy_vectors[player_id][played_actions[player_id]]
        Traveltimes = np.dot(X_i.T, unit_times )

    return Traveltimes


    
def Plot_Network(Network  , congestions  , cmap):
    assert len(congestions) == Network.number_of_edges()

    figure = plt.figure(figsize=(5,8),dpi = 100)

    edge_colors = congestions
    M = len(congestions)
    edge_alphas = np.ones(M)

    node_colors = ['black' for i in range(24)]
    node_size = 50*np.ones(24)

    current_dir = os.getcwd()
    os.chdir("../")
    reader = pandas.read_csv("SiouxFallsNet/SiouxFalls_net.csv")
    init_nodes = reader['Init_node'].values
    term_nodes = reader['Term_node'].values
    lengths = reader['Length'].values
    os.chdir(current_dir)

    for e in range(len(edge_colors)):
        idx1 = np.where(init_nodes == term_nodes[e])
        idx2 = np.where(term_nodes == init_nodes[e])
        idx_edge_same_nodes = int( np.intersect1d(idx1,idx2))
        max_color = np.maximum(edge_colors[e], edge_colors[idx_edge_same_nodes])
        edge_colors[e] = np.float(max_color)

    edge_widths = 14
    nodes = nx.draw_networkx_nodes(Network, nx.get_node_attributes(Network, 'pos'), node_size=node_size, node_color= node_colors)
    edges = nx.draw_networkx_edges(Network, nx.get_node_attributes(Network, 'pos'), node_size=node_size, arrowstyle='-',
                                   arrowsize=10, edge_color=edge_colors,
                                   edge_cmap=cmap, edge_vmin = -20 ,edge_vmax = 90 , width=edge_widths)
    # set alpha value for each edge
    for i in range(M):
        edges[i].set_alpha(edge_alphas[i])

    ax = plt.gca()
    ax.set_axis_off()
    return figure