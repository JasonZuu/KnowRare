import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
import numpy as np

from configs.algo_config import TuckERConfig


class TuckERModel(torch.nn.Module):
    def __init__(self, d1, d2, config:TuckERConfig):
        '''
        params:
            d1: embedding dimension
            d2: relation dimension
            kwargs: input_dropout, hidden_dropout1, hidden_dropout
        '''
        super().__init__()

        self.E = torch.nn.Embedding(config.entity_num, d1)
        self.R = torch.nn.Embedding(config.relation_num, d2)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)), 
                                    dtype=torch.float, requires_grad=True))

        self.input_dropout = torch.nn.Dropout(config.input_dropout)
        self.hidden_dropout1 = torch.nn.Dropout(config.hidden_dropout1)
        self.hidden_dropout2 = torch.nn.Dropout(config.hidden_dropout2)
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)
        self.init()
        

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, h_idx, r_idx):
        h = self.E(h_idx)
        x = self.bn0(h)
        x = self.input_dropout(x)
        x = x.view(-1, 1, h.size(1))

        r = self.R(r_idx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, h.size(1), h.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat) 
        x = x.view(-1, h.size(1))      
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1,0))
        pred = torch.sigmoid(x)
        return pred
    
    def get_node_embedding(self):
        return self.E.weight.detach().cpu().numpy()
    

def test_tucker_model():
    config = TuckERConfig()
    model = TuckERModel(config.d, config.r, config)
    model.to(config.device)
    h_idx = torch.tensor([1, 2, 3], dtype=torch.long).to(config.device)
    r_idx = torch.tensor([1, 2, 0], dtype=torch.long).to(config.device)
    pred = model(h_idx, r_idx)
    print(pred)


if __name__ == "__main__":
    test_tucker_model()
