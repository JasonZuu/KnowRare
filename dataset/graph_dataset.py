import networkx as nx
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from configs.dataset_config import MIMICGraphDatasetConfig


class TuckERDataset(Dataset):
    def __init__(self, 
                 graph: nx.MultiGraph, 
                 set_type='train', 
                 top_ratio=1.0,
                 reverse_edge=True,
                 config=MIMICGraphDatasetConfig):
        """
        Custom graph dataset that returns training or test edges based on set_type.

        Parameters:
        - graph: NetworkX graph
        - set_type: 'train' or 'test'
        - top_ratio: Sampling ratio, keeping the top top_ratio% of edges for each relation
        """
        self.graph = graph
        self.set_type = set_type
        self.top_ratio = top_ratio
        self.reverse_edge = reverse_edge

        self.nodes = list(graph.nodes())
        self.N = len(self.nodes)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        self.relations = config.relations
        self.relation_to_idx = {relation: idx for idx, relation in enumerate(self.relations)}

        self.random_seed = config.random_seed
        self.test_size = config.test_size
        
        # Retrieve all edges and their attributes
        edges = list(graph.edges(data=True, keys=True))
        self.X, self.y = self._load_data(edges)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]  # list
        y_idx = [self.node_to_idx[_y] for _y in y]

        # Convert y to one-hot encoding
        y_onehot = torch.zeros(self.N)
        y_onehot[y_idx] = 1
        return self.node_to_idx[X[0]], self.relation_to_idx[X[1]], y_onehot

    def _load_data(self, edges):
        # Group edges by relation type
        relation_edges = {}
        for edge in edges:
            h, t, r = edge[0], edge[1], edge[2]
            if r not in relation_edges:
                relation_edges[r] = []
            relation_edges[r].append(edge)
        relation_edges = {r: sorted(edges, key=lambda x: x[3]['weight'], reverse=True) for r, edges in relation_edges.items()}
        filtered_edges = {r: edges[:int(len(edges) * self.top_ratio)] for r, edges in relation_edges.items()}
        edges = [edge for edges in filtered_edges.values() for edge in edges]

        # Add reverse edges
        if self.reverse_edge:
            edges += [(edge[1], edge[0], edge[2], {'weight': edge[3]['weight']}) for edge in edges]

        # Split edge set (ensuring consistency across runs)
        train_edges, test_edges = train_test_split(edges, test_size=self.test_size, random_state=self.random_seed)

        # Set different edge sets based on set_type
        if self.set_type == 'train':
            self.edges = train_edges
        elif self.set_type == 'test':
            self.edges = test_edges

        # Convert edges into X=(h,r) and y=(t)
        X = [(edge[0], edge[2]) for edge in self.edges]
        y = [edge[1] for edge in self.edges]
        
        # Merge y for the same X
        X_y = {}
        for i_edge in range(len(X)):
            if X[i_edge] not in X_y:
                X_y[X[i_edge]] = []
            X_y[X[i_edge]].append(y[i_edge])
        
        X = list(X_y.keys())
        y = [X_y[_X] for _X in X]
        return X, y

    def get_N(self):
        return self.N
    
    def get_idx_to_node(self):
        return self.idx_to_node
    
    def get_edge_top_ratio(self):
        return self.top_ratio
