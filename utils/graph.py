import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_graph(node_list):
    G = nx.Graph()
    G.add_nodes_from(node_list)
    return G


def create_multi_graph(node_list):
    G = nx.MultiGraph()
    G.add_nodes_from(node_list)
    return G


def add_or_update_edge(G, node1, node2):
    assert G.has_node(node1), "node1 does not exist"
    assert G.has_node(node2), "node2 does not exist"

    if G.has_edge(node1, node2):
        # if edge exists, increase the weight by 1
        G[node1][node2]['weight'] += 1
    else:
        # if edge does not exist, add edge with weight 1
        G.add_edge(node1, node2, weight=1)

    return G


def update_node_cooccurence_attr(G, node, neighbor_node):
    assert G.has_node(node), "node does not exist"
    assert G.has_node(neighbor_node), "neighbor_node does not exist"
    
    attr_dict = G.nodes[node]
    if 'cooccurence' not in attr_dict:
        attr_dict['cooccurence'] = {}
    
    if neighbor_node in attr_dict['cooccurence']:
        attr_dict['cooccurence'][neighbor_node] += 1
    else:
        attr_dict['cooccurence'][neighbor_node] = 1
    
    return G


def get_icd9_info_for_Gu_mimic(demo_df, ts_df, icd9_codes, target_icd9_code=None):
    icd9_info_dict = {}

    for icd9_code in icd9_codes:
        if target_icd9_code is not None and icd9_code == target_icd9_code:
            hadm_ids = demo_df[(demo_df.icd9_code == icd9_code) & (demo_df.finetune == 1)].hadm_id.unique()
        else:
            hadm_ids = demo_df[demo_df.icd9_code == icd9_code].hadm_id.unique()

        icd9_features = []
        for hadm_id in hadm_ids:
            features = ts_df[ts_df.hadm_id == hadm_id].iloc[:, 3:]
            icd9_features.append(features)

        # Transform the list of dataframes into a single dataframe
        icd9_features_df = pd.concat(icd9_features)

        # Calculate mean, std, and missing ratio for each column
        mean_values = icd9_features_df.mean().values
        std_values = icd9_features_df.std().values
        missing_ratio = icd9_features_df.isnull().mean().values  # get the ratio of missing values in each column

        # Store the results in a numpy array (3, len(features))
        icd9_info = np.array([mean_values, std_values, missing_ratio])
        icd9_info = np.nan_to_num(icd9_info, nan=0) # replace NaN with 0
        icd9_info_dict[icd9_code] = icd9_info
    
    return icd9_info_dict


def get_icd9_info_for_Gu_eicu(demo_df, ts_df, icd9_codes, target_icd9_code=None):
    icd9_info_dict = {}

    for icd9_code in icd9_codes:
        if target_icd9_code is not None and icd9_code == target_icd9_code:
            stay_ids = demo_df[(demo_df.icd9_code == icd9_code) & (demo_df.finetune == 1)].stay_id.unique()
        else:
            stay_ids = demo_df[demo_df.icd9_code == icd9_code].stay_id.unique()

        icd9_features = []
        for stay_id in stay_ids:
            features = ts_df[ts_df['stay_id'] == stay_id].iloc[:, 3:]
            icd9_features.append(features)

        # Transform the list of dataframes into a single dataframe
        icd9_features_df = pd.concat(icd9_features)

        # Calculate mean, std, and missing ratio for each column
        mean_values = icd9_features_df.mean().values
        std_values = icd9_features_df.std().values
        missing_ratio = icd9_features_df.isnull().mean().values  # get the ratio of missing values in each column

        # Store the results in a numpy array (3, len(features))
        icd9_info = np.array([mean_values, std_values, missing_ratio])
        icd9_info = np.nan_to_num(icd9_info, nan=0) # replace NaN with 0
        icd9_info_dict[icd9_code] = icd9_info
    
    return icd9_info_dict


def jaccard_similarity(sequence1, sequence2):
    set1 = set(sequence1)
    set2 = set(sequence2)
    
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    similarity = len(intersection) / len(union)
    return similarity


def draw_graph(G, node_num=20):
    nodes = list(G.nodes())

    # Create a subgraph using the first node_num nodes
    sub_G = G.subgraph(nodes[:node_num])

    # Normalize edge weights to range from 0.1 to 10 for better visualization
    sub_G_weights = [G[u][v]['weight'] for u, v in sub_G.edges()]
    sub_G_weights = np.interp(sub_G_weights, (min(sub_G_weights), max(sub_G_weights)), (0.1, 10))

    # Use the spring layout
    pos = nx.spring_layout(sub_G)

    # Create a color mapping based on node degree for visualization
    node_degree = dict(nx.degree(sub_G))
    colors = [node_degree[n] for n in sub_G.nodes()]
    cmap = plt.get_cmap('viridis')  # You can change the colormap as preferred
    norm = plt.Normalize(min(colors), max(colors))
    node_colors = [cmap(norm(degree)) for degree in colors]

    plt.figure(figsize=(12, 9))
    plt.axis('off')

    # Set the plot background color to black
    plt.gca().set_facecolor('black')

    # Draw nodes without displaying labels
    nx.draw_networkx_nodes(sub_G, pos, node_color=node_colors, node_size=400, cmap=cmap, vmin=min(colors), vmax=max(colors))
    nx.draw_networkx_edges(sub_G, pos, width=sub_G_weights, edge_color='gray')

    plt.show()

    

def draw_multi_graph(G, node_num=20):
    nodes = list(G.nodes())

    # Create a subgraph using the first node_num nodes
    sub_G = G.subgraph(nodes[:node_num])

    # Normalize edge weights to range from 0.1 to 10 for better visualization
    sub_G_weights = [G[u][v][key].get('weight', 1) for u, v, key in sub_G.edges(keys=True)]
    sub_G_weights = np.interp(sub_G_weights, (min(sub_G_weights), max(sub_G_weights)), (0.1, 10))

    # Use the spring layout
    pos = nx.spring_layout(sub_G)

    # Create a color mapping based on node degree for visualization
    node_degree = dict(nx.degree(sub_G))
    colors = [node_degree[n] for n in sub_G.nodes()]
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(min(colors), max(colors))
    node_colors = [cmap(norm(degree)) for degree in colors]

    plt.figure(figsize=(12, 9))
    plt.axis('off')
    plt.gca().set_facecolor('black')  # Set background color to black

    # Draw nodes
    nx.draw_networkx_nodes(sub_G, pos, node_color=node_colors, node_size=400, cmap=cmap, vmin=min(colors), vmax=max(colors))

    # Draw multiple edges
    for (u, v, key), width in zip(sub_G.edges(keys=True), sub_G_weights):
        # Add random offsets to distinguish multiple edges
        offset = np.random.normal(scale=0.02, size=2)
        pos_offset = {k: v + offset for k, v in pos.items()}

        # Set different colors
        edge_color = cmap(norm(node_degree[u])) if 'color' not in G[u][v][key] else G[u][v][key]['color']

        # Draw edges
        nx.draw_networkx_edges(sub_G, pos_offset, edgelist=[(u, v)], width=width, edge_color=edge_color)

    plt.show()
