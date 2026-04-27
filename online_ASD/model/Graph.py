import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm, SAGEConv, EdgeConv
from torch_geometric.utils.dropout import dropout_adj
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.utils import remove_self_loops

class Audio_Add_Visual_GNN(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')  

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, xx=x)
        return out

    def message(self,xx_i, xx_j):
        
        return xx_j + xx_i

class Audio_Weight_Add_Visual_GNN(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')  

    def forward(self, x, weight, edge_index):
        out = self.propagate(edge_index, xx=x, weight=weight)
        return out

    def message(self,xx_i, xx_j, weight_j):
        return weight_j*xx_j + xx_i

class Visual_Add_Audio_GNN(MessagePassing):
    def __init__(self):
        super().__init__(aggr='max')  

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, xx=x)
        return out

    def message(self,xx_i, xx_j):
        return xx_i + xx_j

class GraphASD(torch.nn.Module):
    def __init__(self, channels, feature_dim=1024, dropout=0, dropout_a=0):
        self.channels = channels
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.dropout_a = dropout_a
        super(GraphASD, self).__init__()

        self.layer011 = nn.Linear(self.feature_dim, self.channels[0])
        self.layer012 = nn.Linear(self.feature_dim, self.channels[0])
       
        self.audio_to_visual_frontend_gnn = Audio_Add_Visual_GNN() 
        self.audio_to_visual_frontend_gnn_with_weight = Audio_Weight_Add_Visual_GNN() 

        self.visual_layer11 = EdgeConv(nn.Sequential(nn.Linear(2*self.channels[0], self.channels[0]), nn.ReLU(), nn.Linear(self.channels[0], self.channels[0])))
        self.visual_layer12 = EdgeConv(nn.Sequential(nn.Linear(2*self.channels[0], self.channels[0]), nn.ReLU(), nn.Linear(self.channels[0], self.channels[0])))
        self.visual_layer13 = EdgeConv(nn.Sequential(nn.Linear(2*self.channels[0], self.channels[0]), nn.ReLU(), nn.Linear(self.channels[0], self.channels[0])))
        self.visual_layer21 = SAGEConv(self.channels[0], self.channels[1])
        
        self.visual_to_audio_frontend_gnn = Visual_Add_Audio_GNN()
        self.audio_layer13 = EdgeConv(nn.Sequential(nn.Linear(2*self.channels[0], self.channels[0]), nn.ReLU(), nn.Linear(self.channels[0], self.channels[0])))
        self.fc_audio = nn.Linear(self.channels[0], 1)
        
    def forward(self, x_visual, x_audio, edge_index, edge_attr, speakers):
        device = x_visual.device
        
        x_visual = self.layer011(x_visual)          # clip*speakers x len x dim
        visual_range = x_visual.shape[0] * x_visual.shape[1]
        visual_len = x_visual.shape[1]
        x_audio = self.layer012(x_audio)            # clip x len x dim
        x = torch.cat((x_visual, x_audio), dim=0)   # clip*(speakers+1) x len x dim
        x = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2]))    # clip*(speakers+1)*len x dim
        main_visual_idx = torch.arange(0, visual_range, speakers*visual_len).unsqueeze(1) + torch.arange(visual_len)
        main_visual_idx = torch.reshape(main_visual_idx,(1,-1)).squeeze(0)
        # -----------------------------------------------------------------------------
        #               audio pass
        edge_index_audio = edge_index[:, edge_attr==-2]
        edge_index_vpa = edge_index[:, edge_attr==-3]
        x_audio = self.visual_to_audio_frontend_gnn(x, edge_index_vpa)
        x_audio = F.relu(x_audio)
        a_res = self.audio_layer13(x_audio, edge_index_audio)
        a_res = F.relu(a_res)
        a_res = self.fc_audio(a_res)
        # -----------------------------------------------------------------------------
        
        # -----------------------------------------------------------------------------
        #               visual pass
        edge_index1 = edge_index[:, (edge_attr>=0)&(edge_attr<=1)]
        edge_index2 = edge_index[:, (edge_attr<=0)&(edge_attr>=-1)]
        edge_index3 = edge_index[:, abs(edge_attr)<=1]
        edge_index_acv = edge_index[:, edge_attr==3]

        x_visual = self.audio_to_visual_frontend_gnn_with_weight(x, a_res, edge_index_acv)
        # x_visual = self.audio_to_visual_frontend_gnn(x, edge_index_acv)
        x_visual = F.relu(x_visual)
        
        edge_index1m, _ = dropout_adj(edge_index=edge_index1, p=self.dropout_a, training=self.training)
        x1 = self.visual_layer11(x_visual, edge_index1m)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x1 = self.visual_layer21(x1, edge_index1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        edge_index2m, _ = dropout_adj(edge_index=edge_index2, p=self.dropout_a, training=self.training)
        x2 = self.visual_layer12(x_visual, edge_index2m)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x2 = self.visual_layer21(x2, edge_index2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        edge_index3m, _ = dropout_adj(edge_index=edge_index3, p=self.dropout_a, training=self.training)
        x3 = self.visual_layer13(x_visual, edge_index3m)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, p=self.dropout, training=self.training)
        x3 = self.visual_layer21(x3, edge_index3)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, p=self.dropout, training=self.training)
        x = x1 + x2 + x3
        
        v_res = x[main_visual_idx,:]
        
        return v_res
        # -----------------------------------------------------------------------------