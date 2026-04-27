import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, SAGEConv, DynamicEdgeConv, EdgeConv,GCNConv, GATConv,GraphConv, SGConv, XConv
from torch_geometric.utils.dropout import dropout_adj
from torch_geometric.utils import add_self_loops, remove_self_loops, to_dense_adj
from torch_geometric.utils import dropout_adj, sort_edge_index
import random
from torch.nn import init
import numpy as np 
from model.fusion import *
import ipdb
from torch_geometric.nn import MessagePassing
# seed = 123
# torch.manual_seed(seed)
# torch.backends.cudnn.deterministic = True  
# torch.backends.cudnn.benchmark = False  

def mean_normalize(x, dim=0):
    assert (dim < x.dim())
    mean = torch.mean(x,dim=dim, keepdim=True)
    return x - mean

def l2_normalize(x, dim=0, eps=1e-12):
    assert (dim < x.dim())
    norm = torch.norm(x, 2, dim, keepdim=True)
    return x / (norm + eps)

def feat_normalize(x):
    mean = torch.mean(x,dim=0, keepdim=True)
    x = x - mean
    norm = torch.norm(x, 2, dim=1, keepdim=True)
    


def l2_loss(x):
    norm = torch.norm(x, 2)
    return norm**2



class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, input_dim, d_model, d_k, d_v, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(input_dim, h * d_k)
        self.fc_k = nn.Linear(input_dim, h * d_k)
        self.fc_v = nn.Linear(input_dim, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class AttentionLayer(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, input_dim, d_k, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(AttentionLayer, self).__init__()
        self.fc_q = nn.Linear(input_dim, d_k)
        self.fc_k = nn.Linear(input_dim, d_k)
        self.dropout=nn.Dropout(dropout)

        self.d_k = d_k

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        batch, n, input_dim = queries.shape
        
        q = self.fc_q(queries).permute(0, 1, 2)  # (batch, n, d_k)
        k = self.fc_k(keys).permute(0, 2, 1)     # (batch, d_k, n)
        v = values                               # (batch, n, dim)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (batch, n, n)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v)               # (batch, n, dim)
        out = out[:,0,:]
        
        return out

class AttentionLayer2(nn.Module):

    def __init__(self, input_dim, dropout=.1):
        
        super(AttentionLayer2, self).__init__()
        self.fc_q = nn.Linear(input_dim, 1)
        self.fc_att = nn.Sequential(nn.Linear(8, 8),nn.ReLU(), nn.Linear(8,8))
        self.dropout=nn.Dropout(dropout)
        
    def forward(self, queries, values):
        
        batch, n, input_dim = queries.shape
        
        q = self.fc_q(queries)                  # batch x num x features
        q = q.view(batch, -1)                   # batch x num*features
        att = self.fc_att(q).unsqueeze(2)        # batch x num x 1
        # att=self.dropout(att)                   # F.dropout(x, p=0.9, training=self.training)
        att = F.sigmoid(att)
        # att = F.relu(att)
        att = F.dropout(att, p=0.2, training=self.training)
        
        out = att * values                      # batch x num x input_dim
        out = torch.sum(out, dim=1)             # batch x input_dim
        
        return out

class SimpleFC(torch.nn.Module):
    def __init__(self, channels, feature_dim=1024, dropout=0, dropout_a=0, da_true=False):
        self.channels = channels
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.dropout_a = dropout_a
        self.da_true = da_true
        super(SimpleFC, self).__init__()

        self.fc1 = nn.Linear(self.feature_dim//2, 1)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x_face = x[:,:self.feature_dim//2]
        x_background = x[:,self.feature_dim//2:]
        x = self.fc1(x_face+x_background)
        x = torch.sigmoid(x)
        return x

class feat_trans(torch.nn.Module):
    def __init__(self, feat_dim=128, speaker_dim=16, spatial_dim=16, out_dim=64):
        super(feat_trans, self).__init__()

        self.fc1 = nn.Linear(feat_dim, out_dim)
        self.fc2 = nn.Linear(feat_dim+speaker_dim+spatial_dim, out_dim)
        self.layer1 = EdgeConv(nn.Sequential(nn.Linear(2*out_dim, 2)))
        
    def forward(self, feat, speaker_feat=None, spatial_feat=None, index=None):
        x1 = self.fc1(feat)
        if speaker_feat!=None and spatial_feat!=None:
            x2 = self.fc2(torch.cat((feat, speaker_feat, spatial_feat),dim=1))
            x = x1 + x2
        else:
            x = x1
        
        x = self.layer1(x, index)
        
        return x

class SPELL_bak0827(torch.nn.Module):
    def __init__(self, channels, feature_dim=1024, dropout=0, dropout_a=0, da_true=False):
        self.channels = channels
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.dropout_a = dropout_a
        self.da_true = da_true
        super(SPELL, self).__init__()
        self.feat_dim = 128
        
        self.max_speaker = 3
        speaker_dim=16
        spatial_dim=16
        self.speaker_fc = nn.Linear(self.max_speaker, speaker_dim)
        self.spatial_fc = nn.Linear(4, spatial_dim)
        
        # self.layer_frontend1 = nn.Linear(128, 64)
        # self.layer_frontend2 = nn.Linear(128+32, 64)
        
        self.face_trans = feat_trans(self.feat_dim, speaker_dim=speaker_dim, spatial_dim=spatial_dim, out_dim=64)
        self.face_body_trans = feat_trans(self.feat_dim, speaker_dim=speaker_dim, spatial_dim=spatial_dim, out_dim=64)
        self.face_large_trans = feat_trans(self.feat_dim, speaker_dim=speaker_dim, spatial_dim=spatial_dim, out_dim=64)
        self.background_trans = feat_trans(self.feat_dim, speaker_dim=speaker_dim, spatial_dim=spatial_dim, out_dim=64)
        self.face_small_trans = feat_trans(self.feat_dim, speaker_dim=speaker_dim, spatial_dim=spatial_dim, out_dim=64)
        self.face_body_large_trans = feat_trans(self.feat_dim, speaker_dim=speaker_dim, spatial_dim=spatial_dim, out_dim=64)
        
        
        self.layer1 = EdgeConv(nn.Sequential(nn.Linear(2*64, 2)))
        self.layer2 = SAGEConv(64, 2)
        
        self.fc = nn.Linear(64, 2)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        edge_index1 = edge_index[:, (edge_attr == 111) | (edge_attr == 0)] # co-speaker in one frame 
        
        x_face_av = x[:,:self.feat_dim]
        x_face_body_av = x[:,self.feat_dim:2*self.feat_dim]
        x_face_large_av = x[:,2*self.feat_dim:3*self.feat_dim]
        x_background_av = x[:,3*self.feat_dim:4*self.feat_dim]
        x_face_small_av = x[:,4*self.feat_dim:5*self.feat_dim]
        x_face_down_av = x[:,5*self.feat_dim:6*self.feat_dim]
        x_face_body_large_av = x[:,6*self.feat_dim:7*self.feat_dim]

        speaker_num_feat = x[:, -1] - 1
        speaker_num_feat = torch.clamp(speaker_num_feat, max=self.max_speaker-1).to(torch.int64)
        speaker_num_feat = torch.nn.functional.one_hot(speaker_num_feat, num_classes=self.max_speaker).to(torch.float)
        speaker_num_feat = self.speaker_fc(speaker_num_feat)

        spatial_feat = x[:,-5:-1]
        spatial_feat = self.spatial_fc(spatial_feat)
        
        x_face_av = F.dropout(x_face_av, p=0.9, training=self.training)
        x_face_body_av = F.dropout(x_face_body_av, p=0.9, training=self.training)
        x_face_large_av = F.dropout(x_face_large_av, p=0.9, training=self.training)
        x_background_av = F.dropout(x_background_av, p=0.9, training=self.training)
        x_face_small_av = F.dropout(x_face_small_av, p=0.9, training=self.training)
        x_face_down_av = F.dropout(x_face_down_av, p=0.9, training=self.training)
        x_face_body_large_av = F.dropout(x_face_body_large_av, p=0.9, training=self.training)

        x_face_av = self.face_trans(x_face_av, speaker_num_feat, spatial_feat, edge_index1)
        x_face_body_av = self.face_body_trans(x_face_body_av, speaker_num_feat, spatial_feat, edge_index1)
        x = x_face_av + x_face_body_av
        
        # x1 = self.layer1(x, edge_index1)       
        # x2 = self.layer2(x, edge_index1)
        # x = self.fc(x)
        return x 
    
        return x1 + x2

class Add_GNN(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')  

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, xx=x)
        return out

    def message(self,xx_i, xx_j):
        return xx_j


class SPELL_bak0828(torch.nn.Module):
    def __init__(self, channels, feature_dim=1024, dropout=0, dropout_a=0, da_true=False):
        self.channels = channels
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.dropout_a = dropout_a
        self.da_true = da_true
        super(SPELL, self).__init__()
        self.feat_dim = 128
        
        self.max_speaker = 3
        self.speaker_fc = nn.Linear(self.max_speaker, 16)
        
        self.spatial_fc = nn.Linear(4, 16)
        
        self.layer_frontend1 = nn.Linear(128, 64)
        self.layer_frontend2 = nn.Linear(128+32, 64)
        # self.layer_frontend = nn.Sequential(nn.Linear(128+32, 64), nn.ReLU(), nn.Linear(64,64))
        
        self.layer1 = EdgeConv(nn.Sequential(nn.Linear(2*64, 2)))
        self.layer2 = SAGEConv(64, 2)
        
        # self.layer2 = EdgeConv(nn.Sequential(nn.Linear(2*64, 2)))
        
        # self.scale_graph = EdgeConv(nn.Sequential(nn.Linear(2*128, 64)))
        self.scale_graph1 = Add_GNN()
        self.scale_graph2 = GATConv(in_channels=128, out_channels=64, heads=1)
        
    def forward(self, data):
        x, edge_index, edge_attr, pos = data.x, data.edge_index, data.edge_attr, data.pos
        pos = pos.squeeze(1)
        edge_index1 = edge_index[:, (edge_attr == 111) | (edge_attr == 0)] # co-speaker in one frame 
        edge_index2 = edge_index[:, (edge_attr == 111) | (edge_attr == 222)] # same-speaker with different scale 
        
        x = x[:,:128]
        x = F.dropout(x, p=0.8, training=self.training)
        
        x1 = self.scale_graph1(x, edge_index2)
        x1 = self.layer_frontend1(x1)
        x2 = self.scale_graph2(x, edge_index2)
        x = x1 
        
        x1 = self.layer1(x, edge_index1)       
        x2 = self.layer2(x, edge_index1)
        x = x1 + x2
        
        x = x[pos==1,:]
        
        return x



class SPELL_tmp(torch.nn.Module):
    def __init__(self, channels, feature_dim=1024, dropout=0, dropout_a=0, da_true=False):
        self.channels = channels
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.dropout_a = dropout_a
        self.da_true = da_true
        super(SPELL, self).__init__()
        self.feat_dim = 128
        
        self.max_speaker = 3
        self.speaker_fc = nn.Linear(self.max_speaker, 16)
        
        self.spatial_fc = nn.Linear(4, 16)
        
        self.layer_frontend1 = nn.Linear(128, 64)
        self.layer_frontend2 = nn.Linear(128+32, 64)
        
        self.layer1 = EdgeConv(nn.Sequential(nn.Linear(2*64, 2)))
        self.layer2 = SAGEConv(64, 2)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        edge_index1 = edge_index[:, (edge_attr == 111) | (edge_attr == 0)] # co-speaker in one frame 
        # edge_index1 = edge_index[:, (edge_attr == 111) | (abs(edge_attr) <= 0.1)] # co-speaker in one frame 
        # edge_index1 = edge_index[:, (edge_attr == 111) | ((abs(edge_attr) <= 1)&(edge_attr != 0))] # double-time same-speaker in consecutive frame
        # edge_index3 = edge_index[:, (edge_attr == 111) | ((edge_attr <= self.con_time)&(edge_attr > 0))] # back-time same-speaker in consecutive frame
        # edge_index4 = edge_index[:, (edge_attr == 111) | ((edge_attr >= -self.con_time)&(edge_attr < 0))] # forward-time same-speaker in consecutive frame

        x_face_av =             x[:,:self.feat_dim]
        x_face_body_av =        x[:,1*self.feat_dim:2*self.feat_dim]
        x_face_large_av =       x[:,2*self.feat_dim:3*self.feat_dim]
        x_background_av =       x[:,3*self.feat_dim:4*self.feat_dim]
        x_face_small_av =       x[:,4*self.feat_dim:5*self.feat_dim]
        face_right_feat =       x[:,5*self.feat_dim:6*self.feat_dim]
        face_left_feat =        x[:,6*self.feat_dim:7*self.feat_dim]
        face_up_feat =          x[:,7*self.feat_dim:8*self.feat_dim]
        x_face_down_av =        x[:,8*self.feat_dim:9*self.feat_dim]
        x_face_body_large_av =  x[:,9*self.feat_dim:10*self.feat_dim]
        x_full_main_av =        x[:,10*self.feat_dim:11*self.feat_dim]
        x_body_av =             x[:,11*self.feat_dim:12*self.feat_dim]

        speaker_num_feat = x[:, -1] - 1
        speaker_num_feat = torch.clamp(speaker_num_feat, max=self.max_speaker-1).to(torch.int64)
        speaker_num_feat = torch.nn.functional.one_hot(speaker_num_feat, num_classes=self.max_speaker).to(torch.float)
        speaker_num_feat = self.speaker_fc(speaker_num_feat)
        
        spatial_feat = x[:,-5:-1]
        spatial_feat = self.spatial_fc(spatial_feat)
        
        # x_face_av = F.dropout(x_face_av, p=0.8, training=self.training)
        # x_face_body_av = F.dropout(x_face_body_av, p=0.8, training=self.training)
        # x_face_large_av = F.dropout(x_face_large_av, p=0.8, training=self.training)
        # x_background_av = F.dropout(x_background_av, p=0.8, training=self.training)
        # x_face_small_av = F.dropout(x_face_small_av, p=0.8, training=self.training)
        # x_face_down_av = F.dropout(x_face_down_av, p=0.8, training=self.training)
        # x_face_body_large_av = F.dropout(x_face_body_large_av, p=0.8, training=self.training)
        
        # x = x_face_av + x_face_body_av + x_face_large_av + x_face_small_av + x_face_down_av + x_face_body_large_av + x_background_av
        # # x = x_face_av + x_face_body_av + x_face_large_av + x_face_small_av + x_face_down_av + x_background_av      
        # x = self.layer_frontend(x)
        
        x = torch.concat((
                        x_face_av.unsqueeze(1),
                        x_face_body_av.unsqueeze(1),
                        x_face_large_av.unsqueeze(1),
                        x_face_small_av.unsqueeze(1),
                        x_background_av.unsqueeze(1),
                        x_face_body_large_av.unsqueeze(1),
                        face_right_feat.unsqueeze(1),
                        # 0.01*face_left_feat.unsqueeze(1),
                        # 0.01*face_up_feat.unsqueeze(1),
                        x_face_down_av.unsqueeze(1),
                        # x_full_main_av.unsqueeze(1),
                        x_body_av.unsqueeze(1),
                        ),dim=1)        # batch x num x feat     
        x = F.dropout(x, p=0.9, training=self.training)
        x = torch.sum(x, dim=1)
        
        x1 = self.layer_frontend1(x)
        x2 = self.layer_frontend2(torch.cat((x, speaker_num_feat, spatial_feat), dim=1))                               # batch x 64        
        x = x1 + x2
        
        edge_index1, _ = dropout_adj(edge_index=edge_index1, p=0.01, training=self.training)
        x1 = self.layer1(x, edge_index1)       
        x2 = self.layer2(x, edge_index1)
        
        return x1 + x2



class SPELL(torch.nn.Module):
    def __init__(self):
        super(SPELL, self).__init__()
        self.feat_dim = 128
        
        self.max_speaker = 3
        self.speaker_fc = nn.Linear(self.max_speaker, 16)
        
        self.spatial_fc = nn.Linear(4, 16)
        
        self.layer_frontend1 = nn.Linear(128, 64)
        self.layer_frontend2 = nn.Linear(128+32, 64)
         
        self.layer1 = EdgeConv(nn.Sequential(nn.Linear(2*64, 2)))
        self.layer2 = SAGEConv(64, 2)
                    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        edge_index1 = edge_index[:, (edge_attr == 111) | (edge_attr == 0)] # co-speaker in one frame 

        x_face_av =             x[:,:self.feat_dim]
        x_face_body_av =        x[:,1*self.feat_dim:2*self.feat_dim]
        x_face_large_av =       x[:,2*self.feat_dim:3*self.feat_dim]
        x_background_av =       x[:,3*self.feat_dim:4*self.feat_dim]
        x_face_small_av =       x[:,4*self.feat_dim:5*self.feat_dim]
        face_right_feat =       x[:,5*self.feat_dim:6*self.feat_dim]
        x_face_down_av =        x[:,6*self.feat_dim:7*self.feat_dim]
        x_face_body_large_av =  x[:,7*self.feat_dim:8*self.feat_dim]

        speaker_num_feat = x[:, -1] - 1
        speaker_num_feat = torch.clamp(speaker_num_feat, max=self.max_speaker-1).to(torch.int64)
        speaker_num_feat = torch.nn.functional.one_hot(speaker_num_feat, num_classes=self.max_speaker).to(torch.float)
        speaker_num_feat = self.speaker_fc(speaker_num_feat)
        
        spatial_feat = x[:,-5:-1]
        spatial_feat = self.spatial_fc(spatial_feat)
        
        x = torch.concat((
                        x_face_av.unsqueeze(1),
                        x_face_body_av.unsqueeze(1),
                        x_face_large_av.unsqueeze(1),
                        x_face_small_av.unsqueeze(1),
                        x_background_av.unsqueeze(1),
                        x_face_body_large_av.unsqueeze(1),
                        face_right_feat.unsqueeze(1),
                        x_face_down_av.unsqueeze(1),
                        ),dim=1)        # batch x num x feat     
        x = F.dropout(x, p=0.9, training=self.training)
        x = torch.sum(x, dim=1) / 3
        
        x1 = self.layer_frontend1(x)
        x2 = self.layer_frontend2(torch.cat((x, speaker_num_feat, spatial_feat), dim=1))                               # batch x 64        
        x = x1 + x2
        
        edge_index1, _ = dropout_adj(edge_index=edge_index1, p=0.01, training=self.training)
        x1 = self.layer1(x, edge_index1)       
        x2 = self.layer2(x, edge_index1)
        
        return x1 + x2




class SPELL_bak0830_955598_plaut(torch.nn.Module):
    def __init__(self):
        super(SPELL, self).__init__()
        self.feat_dim = 128
        
        self.max_speaker = 3
        self.speaker_fc = nn.Linear(self.max_speaker, 16)
        
        self.spatial_fc = nn.Linear(4, 16)
        
        self.layer_frontend1 = nn.Linear(128, 64)
        self.layer_frontend2 = nn.Linear(128+32, 64)
         
        self.layer1 = EdgeConv(nn.Sequential(nn.Linear(2*64, 2)))
        self.layer2 = SAGEConv(64, 2)
                    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        edge_index1 = edge_index[:, (edge_attr == 111) | (edge_attr == 0)] # co-speaker in one frame 

        x_face_av =             x[:,:self.feat_dim]
        x_face_body_av =        x[:,1*self.feat_dim:2*self.feat_dim]
        x_face_large_av =       x[:,2*self.feat_dim:3*self.feat_dim]
        x_background_av =       x[:,3*self.feat_dim:4*self.feat_dim]
        x_face_small_av =       x[:,4*self.feat_dim:5*self.feat_dim]
        face_right_feat =       x[:,5*self.feat_dim:6*self.feat_dim]
        x_face_down_av =        x[:,6*self.feat_dim:7*self.feat_dim]
        x_face_body_large_av =  x[:,7*self.feat_dim:8*self.feat_dim]
        x_body_av =             x[:,8*self.feat_dim:9*self.feat_dim]

        speaker_num_feat = x[:, -1] - 1
        speaker_num_feat = torch.clamp(speaker_num_feat, max=self.max_speaker-1).to(torch.int64)
        speaker_num_feat = torch.nn.functional.one_hot(speaker_num_feat, num_classes=self.max_speaker).to(torch.float)
        speaker_num_feat = self.speaker_fc(speaker_num_feat)
        
        spatial_feat = x[:,-5:-1]
        spatial_feat = self.spatial_fc(spatial_feat)
        
        x = torch.concat((
                        x_face_av.unsqueeze(1),
                        x_face_body_av.unsqueeze(1),
                        x_face_large_av.unsqueeze(1),
                        x_face_small_av.unsqueeze(1),
                        x_background_av.unsqueeze(1),
                        x_face_body_large_av.unsqueeze(1),
                        face_right_feat.unsqueeze(1),
                        x_face_down_av.unsqueeze(1),
                        # x_body_av.unsqueeze(1),
                        ),dim=1)        # batch x num x feat     
        x = F.dropout(x, p=0.9, training=self.training)
        x = torch.sum(x, dim=1) / 3
        
        x1 = self.layer_frontend1(x)
        x2 = self.layer_frontend2(torch.cat((x, speaker_num_feat, spatial_feat), dim=1))                               # batch x 64        
        x = x1 + x2
        
        edge_index1, _ = dropout_adj(edge_index=edge_index1, p=0.01, training=self.training)
        x1 = self.layer1(x, edge_index1)       
        x2 = self.layer2(x, edge_index1)
        
        return x1 + x2

class SPELL_bak0830_955574_platau_seed0(torch.nn.Module):
    def __init__(self, channels, feature_dim=1024, dropout=0, dropout_a=0, da_true=False):
        self.channels = channels
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.dropout_a = dropout_a
        self.da_true = da_true
        super(SPELL, self).__init__()
        self.feat_dim = 128
        
        self.max_speaker = 3
        self.speaker_fc = nn.Linear(self.max_speaker, 16)
        
        self.spatial_fc = nn.Linear(4, 16)
        
        self.layer_frontend1 = nn.Linear(128, 64)
        self.layer_frontend2 = nn.Linear(128+32, 64)
        # self.layer_frontend2 = nn.Sequential(nn.Linear(128+32, 32), nn.ReLU(), nn.Linear(32,64))
        # self.layer_frontend = nn.Sequential(nn.Linear(128+32, 64), nn.ReLU(), nn.Linear(64,64))
        
        self.layer1 = EdgeConv(nn.Sequential(nn.Linear(2*64, 2)))
        self.layer2 = SAGEConv(64, 2)
        # self.layer2 = SGConv(64,2)
        
        # self.layer2 = EdgeConv(nn.Sequential(nn.Linear(2*64, 2)))
        
        self.final_fc = nn.Linear(4, 2)
        self.confident_fc = nn.Linear(128+16, 128)

        # self.att_layer = AttentionLayer(input_dim=128, d_k=16)
        # self.att_layer = AttentionLayer2(input_dim=128)
        # self.fc = nn.Linear(16, 7)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        edge_index1 = edge_index[:, (edge_attr == 111) | (edge_attr == 0)] # co-speaker in one frame 
        # edge_index1 = edge_index[:, (edge_attr == 111) | (abs(edge_attr) <= 0.1)] # co-speaker in one frame 
        # edge_index1 = edge_index[:, (edge_attr == 111) | ((abs(edge_attr) <= 1)&(edge_attr != 0))] # double-time same-speaker in consecutive frame
        # edge_index3 = edge_index[:, (edge_attr == 111) | ((edge_attr <= self.con_time)&(edge_attr > 0))] # back-time same-speaker in consecutive frame
        # edge_index4 = edge_index[:, (edge_attr == 111) | ((edge_attr >= -self.con_time)&(edge_attr < 0))] # forward-time same-speaker in consecutive frame

        x_face_av =             x[:,:self.feat_dim]
        x_face_body_av =        x[:,1*self.feat_dim:2*self.feat_dim]
        x_face_large_av =       x[:,2*self.feat_dim:3*self.feat_dim]
        x_background_av =       x[:,3*self.feat_dim:4*self.feat_dim]
        x_face_small_av =       x[:,4*self.feat_dim:5*self.feat_dim]
        face_right_feat =       x[:,5*self.feat_dim:6*self.feat_dim]
        face_left_feat =        x[:,6*self.feat_dim:7*self.feat_dim]
        face_up_feat =          x[:,7*self.feat_dim:8*self.feat_dim]
        x_face_down_av =        x[:,8*self.feat_dim:9*self.feat_dim]
        x_face_body_large_av =  x[:,9*self.feat_dim:10*self.feat_dim]
        x_full_main_av =        x[:,10*self.feat_dim:11*self.feat_dim]
        x_body_av =             x[:,11*self.feat_dim:12*self.feat_dim]

        speaker_num_feat = x[:, -1] - 1
        speaker_num_feat = torch.clamp(speaker_num_feat, max=self.max_speaker-1).to(torch.int64)
        speaker_num_feat = torch.nn.functional.one_hot(speaker_num_feat, num_classes=self.max_speaker).to(torch.float)
        speaker_num_feat = self.speaker_fc(speaker_num_feat)
        
        spatial_feat = x[:,-5:-1]
        spatial_feat = self.spatial_fc(spatial_feat)
        
        # x_face_av = F.dropout(x_face_av, p=0.8, training=self.training)
        # x_face_body_av = F.dropout(x_face_body_av, p=0.8, training=self.training)
        # x_face_large_av = F.dropout(x_face_large_av, p=0.8, training=self.training)
        # x_background_av = F.dropout(x_background_av, p=0.8, training=self.training)
        # x_face_small_av = F.dropout(x_face_small_av, p=0.8, training=self.training)
        # x_face_down_av = F.dropout(x_face_down_av, p=0.8, training=self.training)
        # x_face_body_large_av = F.dropout(x_face_body_large_av, p=0.8, training=self.training)
        
        # x = x_face_av + x_face_body_av + x_face_large_av + x_face_small_av + x_face_down_av + x_face_body_large_av + x_background_av
        # # x = x_face_av + x_face_body_av + x_face_large_av + x_face_small_av + x_face_down_av + x_background_av      
        # x = self.layer_frontend(x)
        
        x = torch.concat((
                        x_face_av.unsqueeze(1),
                        x_face_body_av.unsqueeze(1),
                        x_face_large_av.unsqueeze(1),
                        x_face_small_av.unsqueeze(1),
                        x_background_av.unsqueeze(1),
                        x_face_body_large_av.unsqueeze(1),
                        face_right_feat.unsqueeze(1),
                        # 0.01*face_left_feat.unsqueeze(1),
                        # 0.01*face_up_feat.unsqueeze(1),
                        x_face_down_av.unsqueeze(1),
                        # x_full_main_av.unsqueeze(1),
                        x_body_av.unsqueeze(1),
                        ),dim=1)        # batch x num x feat     
        x = F.dropout(x, p=0.9, training=self.training)
        x = torch.sum(x, dim=1)
        
        x1 = self.layer_frontend1(x)
        x2 = self.layer_frontend2(torch.cat((x, speaker_num_feat, spatial_feat), dim=1))                               # batch x 64        
        x = x1 + x2
        # x = F.dropout(x, p=0.9, training=self.training)
        
        edge_index1, _ = dropout_adj(edge_index=edge_index1, p=0.01, training=self.training)
        x1 = self.layer1(x, edge_index1)       
        x2 = self.layer2(x, edge_index1)
        
        return x1 + x2



class SPELL_9555_50_usingPlaut(torch.nn.Module):
    def __init__(self, channels, feature_dim=1024, dropout=0, dropout_a=0, da_true=False):
        self.channels = channels
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.dropout_a = dropout_a
        self.da_true = da_true
        super(SPELL, self).__init__()
        self.feat_dim = 128
        
        self.max_speaker = 3
        self.speaker_fc = nn.Linear(self.max_speaker, 16)
        
        self.spatial_fc = nn.Linear(4, 16)
        
        self.layer_frontend1 = nn.Linear(128, 64)
        self.layer_frontend2 = nn.Linear(128+32, 64)
        # self.layer_frontend2 = nn.Sequential(nn.Linear(128+32, 32), nn.ReLU(), nn.Linear(32,64))
        # self.layer_frontend = nn.Sequential(nn.Linear(128+32, 64), nn.ReLU(), nn.Linear(64,64))
        
        self.layer1 = EdgeConv(nn.Sequential(nn.Linear(2*64, 2)))
        self.layer2 = SAGEConv(64, 2)
        # self.layer2 = SGConv(64,2)
        
        # self.layer2 = EdgeConv(nn.Sequential(nn.Linear(2*64, 2)))
        
        self.final_fc = nn.Linear(4, 2)
        self.confident_fc = nn.Linear(128+16, 128)

        # self.att_layer = AttentionLayer(input_dim=128, d_k=16)
        # self.att_layer = AttentionLayer2(input_dim=128)
        # self.fc = nn.Linear(16, 7)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        edge_index1 = edge_index[:, (edge_attr == 111) | (edge_attr == 0)] # co-speaker in one frame 
        # edge_index1 = edge_index[:, (edge_attr == 111) | (abs(edge_attr) <= 0.1)] # co-speaker in one frame 
        # edge_index1 = edge_index[:, (edge_attr == 111) | ((abs(edge_attr) <= 1)&(edge_attr != 0))] # double-time same-speaker in consecutive frame
        # edge_index3 = edge_index[:, (edge_attr == 111) | ((edge_attr <= self.con_time)&(edge_attr > 0))] # back-time same-speaker in consecutive frame
        # edge_index4 = edge_index[:, (edge_attr == 111) | ((edge_attr >= -self.con_time)&(edge_attr < 0))] # forward-time same-speaker in consecutive frame

        x_face_av =             x[:,:self.feat_dim]
        x_face_body_av =        x[:,1*self.feat_dim:2*self.feat_dim]
        x_face_large_av =       x[:,2*self.feat_dim:3*self.feat_dim]
        x_background_av =       x[:,3*self.feat_dim:4*self.feat_dim]
        x_face_small_av =       x[:,4*self.feat_dim:5*self.feat_dim]
        face_right_feat =       x[:,5*self.feat_dim:6*self.feat_dim]
        face_left_feat =        x[:,6*self.feat_dim:7*self.feat_dim]
        face_up_feat =          x[:,7*self.feat_dim:8*self.feat_dim]
        x_face_down_av =        x[:,8*self.feat_dim:9*self.feat_dim]
        x_face_body_large_av =  x[:,9*self.feat_dim:10*self.feat_dim]
        x_full_main_av =        x[:,10*self.feat_dim:11*self.feat_dim]

        speaker_num_feat = x[:, -1] - 1
        speaker_num_feat = torch.clamp(speaker_num_feat, max=self.max_speaker-1).to(torch.int64)
        speaker_num_feat = torch.nn.functional.one_hot(speaker_num_feat, num_classes=self.max_speaker).to(torch.float)
        speaker_num_feat = self.speaker_fc(speaker_num_feat)
        
        spatial_feat = x[:,-5:-1]
        spatial_feat = self.spatial_fc(spatial_feat)
        
        # x_face_av = F.dropout(x_face_av, p=0.8, training=self.training)
        # x_face_body_av = F.dropout(x_face_body_av, p=0.8, training=self.training)
        # x_face_large_av = F.dropout(x_face_large_av, p=0.8, training=self.training)
        # x_background_av = F.dropout(x_background_av, p=0.8, training=self.training)
        # x_face_small_av = F.dropout(x_face_small_av, p=0.8, training=self.training)
        # x_face_down_av = F.dropout(x_face_down_av, p=0.8, training=self.training)
        # x_face_body_large_av = F.dropout(x_face_body_large_av, p=0.8, training=self.training)
        
        # x = x_face_av + x_face_body_av + x_face_large_av + x_face_small_av + x_face_down_av + x_face_body_large_av + x_background_av
        # # x = x_face_av + x_face_body_av + x_face_large_av + x_face_small_av + x_face_down_av + x_background_av      
        # x = self.layer_frontend(x)
        
        x = torch.concat((
                        x_face_av.unsqueeze(1),
                        x_face_body_av.unsqueeze(1),
                        x_face_large_av.unsqueeze(1),
                        x_face_small_av.unsqueeze(1),
                        x_background_av.unsqueeze(1),
                        x_face_body_large_av.unsqueeze(1),
                        face_right_feat.unsqueeze(1),
                        # 0.01*face_left_feat.unsqueeze(1),
                        # 0.01*face_up_feat.unsqueeze(1),
                        x_face_down_av.unsqueeze(1),
                        0.01*x_full_main_av.unsqueeze(1),
                        ),dim=1)        # batch x num x feat     
        x = F.dropout(x, p=0.9, training=self.training)
        x = torch.sum(x, dim=1)
        
        x1 = self.layer_frontend1(x)
        x2 = self.layer_frontend2(torch.cat((x, speaker_num_feat, spatial_feat), dim=1))                               # batch x 64        
        x = x1 + x2
        # x = F.dropout(x, p=0.9, training=self.training)
        
        edge_index1, _ = dropout_adj(edge_index=edge_index1, p=0.01, training=self.training)
        x1 = self.layer1(x, edge_index1)       
        x2 = self.layer2(x, edge_index1)
        
        return x1 + x2




class SPELL_bak0829_955375_47(torch.nn.Module):
    def __init__(self, channels, feature_dim=1024, dropout=0, dropout_a=0, da_true=False):
        self.channels = channels
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.dropout_a = dropout_a
        self.da_true = da_true
        super(SPELL, self).__init__()
        self.feat_dim = 128
        
        self.max_speaker = 3
        self.speaker_fc = nn.Linear(self.max_speaker, 16)
        
        self.spatial_fc = nn.Linear(4, 16)
        
        self.layer_frontend1 = nn.Linear(128, 64)
        self.layer_frontend2 = nn.Linear(128+32, 64)
        # self.layer_frontend2 = nn.Sequential(nn.Linear(128+32, 32), nn.ReLU(), nn.Linear(32,64))
        # self.layer_frontend = nn.Sequential(nn.Linear(128+32, 64), nn.ReLU(), nn.Linear(64,64))
        
        self.layer1 = EdgeConv(nn.Sequential(nn.Linear(2*64, 2)))
        self.layer2 = SAGEConv(64, 2)
        # self.layer2 = SGConv(64,2)
        
        # self.layer2 = EdgeConv(nn.Sequential(nn.Linear(2*64, 2)))
        
        self.final_fc = nn.Linear(4, 2)
        self.confident_fc = nn.Linear(128+16, 128)

        # self.att_layer = AttentionLayer(input_dim=128, d_k=16)
        # self.att_layer = AttentionLayer2(input_dim=128)
        # self.fc = nn.Linear(16, 7)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        edge_index1 = edge_index[:, (edge_attr == 111) | (edge_attr == 0)] # co-speaker in one frame 
        # edge_index1 = edge_index[:, (edge_attr == 111) | (abs(edge_attr) <= 0.1)] # co-speaker in one frame 
        # edge_index1 = edge_index[:, (edge_attr == 111) | ((abs(edge_attr) <= 1)&(edge_attr != 0))] # double-time same-speaker in consecutive frame
        # edge_index3 = edge_index[:, (edge_attr == 111) | ((edge_attr <= self.con_time)&(edge_attr > 0))] # back-time same-speaker in consecutive frame
        # edge_index4 = edge_index[:, (edge_attr == 111) | ((edge_attr >= -self.con_time)&(edge_attr < 0))] # forward-time same-speaker in consecutive frame

        x_face_av =             x[:,:self.feat_dim]
        x_face_body_av =        x[:,1*self.feat_dim:2*self.feat_dim]
        x_face_large_av =       x[:,2*self.feat_dim:3*self.feat_dim]
        x_background_av =       x[:,3*self.feat_dim:4*self.feat_dim]
        x_face_small_av =       x[:,4*self.feat_dim:5*self.feat_dim]
        face_right_feat =       x[:,5*self.feat_dim:6*self.feat_dim]
        face_left_feat =        x[:,6*self.feat_dim:7*self.feat_dim]
        face_up_feat =          x[:,7*self.feat_dim:8*self.feat_dim]
        x_face_down_av =        x[:,8*self.feat_dim:9*self.feat_dim]
        x_face_body_large_av =  x[:,9*self.feat_dim:10*self.feat_dim]
        x_full_main_av =        x[:,10*self.feat_dim:11*self.feat_dim]

        speaker_num_feat = x[:, -1] - 1
        speaker_num_feat = torch.clamp(speaker_num_feat, max=self.max_speaker-1).to(torch.int64)
        speaker_num_feat = torch.nn.functional.one_hot(speaker_num_feat, num_classes=self.max_speaker).to(torch.float)
        speaker_num_feat = self.speaker_fc(speaker_num_feat)
        
        spatial_feat = x[:,-5:-1]
        spatial_feat = self.spatial_fc(spatial_feat)
        
        # x_face_av = F.dropout(x_face_av, p=0.8, training=self.training)
        # x_face_body_av = F.dropout(x_face_body_av, p=0.8, training=self.training)
        # x_face_large_av = F.dropout(x_face_large_av, p=0.8, training=self.training)
        # x_background_av = F.dropout(x_background_av, p=0.8, training=self.training)
        # x_face_small_av = F.dropout(x_face_small_av, p=0.8, training=self.training)
        # x_face_down_av = F.dropout(x_face_down_av, p=0.8, training=self.training)
        # x_face_body_large_av = F.dropout(x_face_body_large_av, p=0.8, training=self.training)
        
        # x = x_face_av + x_face_body_av + x_face_large_av + x_face_small_av + x_face_down_av + x_face_body_large_av + x_background_av
        # # x = x_face_av + x_face_body_av + x_face_large_av + x_face_small_av + x_face_down_av + x_background_av      
        # x = self.layer_frontend(x)
        
        x = torch.concat((
                        x_face_av.unsqueeze(1),
                        x_face_body_av.unsqueeze(1),
                        x_face_large_av.unsqueeze(1),
                        x_face_small_av.unsqueeze(1),
                        x_background_av.unsqueeze(1),
                        x_face_body_large_av.unsqueeze(1),
                        # face_right_feat.unsqueeze(1),
                        # face_left_feat.unsqueeze(1), 
                        # face_up_feat.unsqueeze(1),
                        x_face_down_av.unsqueeze(1),
                        0.01*x_full_main_av.unsqueeze(1),
                        ),dim=1)        # batch x num x feat     
        x = F.dropout(x, p=0.9, training=self.training)
        x = torch.sum(x, dim=1)
        
        x1 = self.layer_frontend1(x)
        x2 = self.layer_frontend2(torch.cat((x, speaker_num_feat, spatial_feat), dim=1))                               # batch x 64        
        x = x1 + x2
        # x = F.dropout(x, p=0.9, training=self.training)
        
        edge_index1, _ = dropout_adj(edge_index=edge_index1, p=0.01, training=self.training)
        x1 = self.layer1(x, edge_index1)       
        x2 = self.layer2(x, edge_index1)
        
        return x1 + x2




class SPELL_bak0829_9521_47(torch.nn.Module):
    def __init__(self, channels, feature_dim=1024, dropout=0, dropout_a=0, da_true=False):
        self.channels = channels
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.dropout_a = dropout_a
        self.da_true = da_true
        super(SPELL, self).__init__()
        self.feat_dim = 128
        
        self.max_speaker = 3
        self.speaker_fc = nn.Linear(self.max_speaker, 16)
        
        self.spatial_fc = nn.Linear(4, 16)
        
        self.layer_frontend1 = nn.Linear(128, 64)
        self.layer_frontend2 = nn.Linear(128+32, 64)
        # self.layer_frontend2 = nn.Sequential(nn.Linear(128+32, 32), nn.ReLU(), nn.Linear(32,64))
        # self.layer_frontend = nn.Sequential(nn.Linear(128+32, 64), nn.ReLU(), nn.Linear(64,64))
        
        self.layer1 = EdgeConv(nn.Sequential(nn.Linear(2*64, 2)))
        self.layer2 = SAGEConv(64, 2)
        # self.layer2 = SGConv(64,2)
        
        # self.layer2 = EdgeConv(nn.Sequential(nn.Linear(2*64, 2)))
        
        self.final_fc = nn.Linear(4, 2)
        self.confident_fc = nn.Linear(128+16, 128)

        # self.att_layer = AttentionLayer(input_dim=128, d_k=16)
        # self.att_layer = AttentionLayer2(input_dim=128)
        # self.fc = nn.Linear(16, 7)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        edge_index1 = edge_index[:, (edge_attr == 111) | (edge_attr == 0)] # co-speaker in one frame 
        # edge_index1 = edge_index[:, (edge_attr == 111) | (abs(edge_attr) <= 0.1)] # co-speaker in one frame 
        # edge_index1 = edge_index[:, (edge_attr == 111) | ((abs(edge_attr) <= 1)&(edge_attr != 0))] # double-time same-speaker in consecutive frame
        # edge_index3 = edge_index[:, (edge_attr == 111) | ((edge_attr <= self.con_time)&(edge_attr > 0))] # back-time same-speaker in consecutive frame
        # edge_index4 = edge_index[:, (edge_attr == 111) | ((edge_attr >= -self.con_time)&(edge_attr < 0))] # forward-time same-speaker in consecutive frame

        x_face_av =             x[:,:self.feat_dim]
        x_face_body_av =        x[:,1*self.feat_dim:2*self.feat_dim]
        x_face_large_av =       x[:,2*self.feat_dim:3*self.feat_dim]
        x_background_av =       x[:,3*self.feat_dim:4*self.feat_dim]
        x_face_small_av =       x[:,4*self.feat_dim:5*self.feat_dim]
        face_right_feat =       x[:,5*self.feat_dim:6*self.feat_dim]
        face_left_feat =        x[:,6*self.feat_dim:7*self.feat_dim]
        face_up_feat =          x[:,7*self.feat_dim:8*self.feat_dim]
        x_face_down_av =        x[:,8*self.feat_dim:9*self.feat_dim]
        x_face_body_large_av =  x[:,9*self.feat_dim:10*self.feat_dim]
        x_full_main_av =        x[:,10*self.feat_dim:11*self.feat_dim]

        speaker_num_feat = x[:, -1] - 1
        speaker_num_feat = torch.clamp(speaker_num_feat, max=self.max_speaker-1).to(torch.int64)
        speaker_num_feat = torch.nn.functional.one_hot(speaker_num_feat, num_classes=self.max_speaker).to(torch.float)
        speaker_num_feat = self.speaker_fc(speaker_num_feat)
        
        spatial_feat = x[:,-5:-1]
        spatial_feat = self.spatial_fc(spatial_feat)
        
        # x_face_av = F.dropout(x_face_av, p=0.8, training=self.training)
        # x_face_body_av = F.dropout(x_face_body_av, p=0.8, training=self.training)
        # x_face_large_av = F.dropout(x_face_large_av, p=0.8, training=self.training)
        # x_background_av = F.dropout(x_background_av, p=0.8, training=self.training)
        # x_face_small_av = F.dropout(x_face_small_av, p=0.8, training=self.training)
        # x_face_down_av = F.dropout(x_face_down_av, p=0.8, training=self.training)
        # x_face_body_large_av = F.dropout(x_face_body_large_av, p=0.8, training=self.training)
        
        # x = x_face_av + x_face_body_av + x_face_large_av + x_face_small_av + x_face_down_av + x_face_body_large_av + x_background_av
        # # x = x_face_av + x_face_body_av + x_face_large_av + x_face_small_av + x_face_down_av + x_background_av      
        # x = self.layer_frontend(x)
        
        x = torch.concat((
                        x_face_av.unsqueeze(1),
                        x_face_body_av.unsqueeze(1),
                        x_face_large_av.unsqueeze(1),
                        x_face_small_av.unsqueeze(1),
                        x_background_av.unsqueeze(1),
                        x_face_body_large_av.unsqueeze(1),
                        # 0.01*face_right_feat.unsqueeze(1),
                        # face_left_feat.unsqueeze(1),
                        # face_up_feat.unsqueeze(1),
                        x_face_down_av.unsqueeze(1),
                        # 0.01*x_full_main_av.unsqueeze(1),
                        ),dim=1)        # batch x num x feat     
        x = F.dropout(x, p=0.9, training=self.training)
        x = torch.sum(x, dim=1)
        
        x1 = self.layer_frontend1(x)
        x2 = self.layer_frontend2(torch.cat((x, speaker_num_feat, spatial_feat), dim=1))                               # batch x 64        
        x = x1 + x2
        # x = F.dropout(x, p=0.9, training=self.training)
        
        edge_index1, _ = dropout_adj(edge_index=edge_index1, p=0.01, training=self.training)
        x1 = self.layer1(x, edge_index1)       
        x2 = self.layer2(x, edge_index1)
        
        return x1 + x2



class SPELL_bak_0821(torch.nn.Module):
    def __init__(self, channels, feature_dim=1024, dropout=0, dropout_a=0, da_true=False):
        self.channels = channels
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.dropout_a = dropout_a
        self.da_true = da_true
        super(SPELL, self).__init__()
        self.feat_dim = 128

        self.layer_frontend = nn.Linear(128, 64)
        self.layer_backend = nn.Linear(64,2)
        # self.layer_frontend = nn.Sequential(nn.Linear(5*128, 2*128), nn.ReLU(), nn.Linear(2*128, 64))
        # self.layer_frontend = EdgeConv(nn.Sequential(nn.Linear(2*128, 64)))
        
        self.layer1 = EdgeConv(nn.Sequential(nn.Linear(2*64, 2)))
        self.layer2 = SAGEConv(64, 2)
        

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        edge_index1 = edge_index[:, (edge_attr == 111) | (edge_attr == 0)] #
        # edge_index1 = edge_index[:, (edge_attr == 111) | (abs(edge_attr) <= 0.5)] #
        # edge_index2 = edge_index[:, (edge_attr == 111) | ((abs(edge_attr) <= 0.1)&(edge_attr != 0))] #95.0135

        x_face_av = x[:,:self.feat_dim]
        x_face_body_av = x[:,self.feat_dim:2*self.feat_dim]
        x_face_large_av = x[:,2*self.feat_dim:3*self.feat_dim]
        x_background_av = x[:,3*self.feat_dim:4*self.feat_dim]
        x_face_small_av = x[:,4*self.feat_dim:5*self.feat_dim]
        x_face_down_av = x[:,5*self.feat_dim:6*self.feat_dim]
        
        
        x_face_av = F.dropout(x_face_av, p=0.6, training=self.training)
        x_face_body_av = F.dropout(x_face_body_av, p=0.6, training=self.training)
        x_face_large_av = F.dropout(x_face_large_av, p=0.6, training=self.training)
        x_background_av = F.dropout(x_background_av, p=0.6, training=self.training)
        x_face_small_av = F.dropout(x_face_small_av, p=0.6, training=self.training)
        x_face_down_av = F.dropout(x_face_down_av, p=0.6, training=self.training)
        
        x = torch.concat((x_face_av.unsqueeze(1),
                       x_face_body_av.unsqueeze(1),
                       x_face_large_av.unsqueeze(1),
                       x_face_small_av.unsqueeze(1),
                    #    x_face_right_av.unsqueeze(1),
                    #    x_face_left_av.unsqueeze(1),
                    #    x_face_up_av.unsqueeze(1),
                    #    x_background_av.unsqueeze(1),
                       x_face_down_av.unsqueeze(1)
                        ),dim=1)        # batch x num x feat
        
        x = torch.sum(x, dim=1)                                  # batch x feat
        x = self.layer_frontend(x)                               # batch x 64        
        # x = F.relu(x)
        # x = self.layer_backend(x)                                # batch x 2
        # return x 
        x1 = self.layer1(x, edge_index1)       
        x2 = self.layer2(x, edge_index1)
        
        # return x11
        return x1 + x2
        # return x1 + x3
        # return x1 + x2 + x4
        # return x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11
        # return x1 + x2
        # return x1
        # return x1 + x2 + x3



class SPELLbak2(torch.nn.Module):
    def __init__(self, channels, feature_dim=1024, dropout=0, dropout_a=0, da_true=False):
        self.channels = channels
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.dropout_a = dropout_a
        self.da_true = da_true
        super(SPELL, self).__init__()
        self.feat_dim = 128

        self.layer_frontend = nn.Linear(128, self.channels[0])
        
        self.layer1 = EdgeConv(nn.Sequential(nn.Linear(2*self.channels[0], 2)))
        self.layer2 = EdgeConv(nn.Sequential(nn.Linear(2*self.channels[0], 2)))
        self.layer3 = EdgeConv(nn.Sequential(nn.Linear(2*self.channels[0], 2)))
        self.layer4 = EdgeConv(nn.Sequential(nn.Linear(2*self.channels[0], 2)))
        self.layer5 = EdgeConv(nn.Sequential(nn.Linear(2*self.channels[0], 2)))
        self.layer6 = EdgeConv(nn.Sequential(nn.Linear(2*self.channels[0], 2)))
        self.layer7 = EdgeConv(nn.Sequential(nn.Linear(2*self.channels[0], 2)))
        self.layer8 = EdgeConv(nn.Sequential(nn.Linear(2*self.channels[0], 2)))
        self.layer9 = EdgeConv(nn.Sequential(nn.Linear(2*self.channels[0], 2)))
        self.layer10 = EdgeConv(nn.Sequential(nn.Linear(2*self.channels[0], 2)))
        self.layer11 = EdgeConv(nn.Sequential(nn.Linear(2*self.channels[0], 2)))
        
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x_face_av = x[:,:self.feat_dim]
        x_face_body_av = x[:,self.feat_dim:2*self.feat_dim]
        x_face_large_av = x[:,2*self.feat_dim:3*self.feat_dim]
        x_background_av = x[:,3*self.feat_dim:4*self.feat_dim]
        x_face_small_av = x[:,4*self.feat_dim:5*self.feat_dim]
        
        x_face_av = F.dropout(x_face_av, p=0.6, training=self.training)
        x_face_body_av = F.dropout(x_face_body_av, p=0.6, training=self.training)
        x_face_large_av = F.dropout(x_face_large_av, p=0.6, training=self.training)
        x_background_av = F.dropout(x_background_av, p=0.6, training=self.training)
        x_face_small_av = F.dropout(x_face_small_av, p=0.6, training=self.training)
        
        
        x = self.layer_frontend(x_face_av+x_face_body_av+x_face_large_av+x_face_small_av)
        x = F.dropout(x, p=0.1, training=self.training)
        
        edge_index1 = edge_index[:, (edge_attr == 111) | (edge_attr == 0)] #
        edge_index2 = edge_index[:, (edge_attr == 111) | ((abs(edge_attr) < 0.05)&(edge_attr != 0))] #95.0135
        edge_index3 = edge_index[:, (edge_attr == 111) | ((abs(edge_attr) >= 0.05)&(abs(edge_attr) < 0.1))]
        edge_index4 = edge_index[:, (edge_attr == 111) | ((abs(edge_attr) >= 0.1)&(abs(edge_attr) < 0.15))]
        edge_index5 = edge_index[:, (edge_attr == 111) | ((abs(edge_attr) >= 0.15)&(abs(edge_attr) < 0.2))]
        edge_index6 = edge_index[:, (edge_attr == 111) | ((abs(edge_attr) >= 0.2)&(abs(edge_attr) < 0.25))]
        edge_index7 = edge_index[:, (edge_attr == 111) | ((abs(edge_attr) >= 0.25)&(abs(edge_attr) < 0.3))]
        edge_index8 = edge_index[:, (edge_attr == 111) | ((abs(edge_attr) >= 0.3)&(abs(edge_attr) < 0.35))]
        edge_index9 = edge_index[:, (edge_attr == 111) | ((abs(edge_attr) >= 0.35)&(abs(edge_attr) < 0.4))]
        edge_index10 = edge_index[:, (edge_attr == 111) | ((abs(edge_attr) >= 0.4)&(abs(edge_attr) < 0.45))]
        edge_index11 = edge_index[:, (edge_attr == 111) | (abs(edge_attr) <= 0.05)]
        
        x1 = self.layer1(x, edge_index1)
        x2 = self.layer2(x, edge_index2)    
        x3 = self.layer3(x, edge_index3)
        x4 = self.layer4(x, edge_index4)
        x5 = self.layer5(x, edge_index5)
        x6 = self.layer6(x, edge_index6)
        x7 = self.layer7(x, edge_index7)
        x8 = self.layer8(x, edge_index8)
        x9 = self.layer9(x, edge_index9)
        x10 = self.layer10(x, edge_index10)
        x11 = self.layer11(x, edge_index11)                          
        
        # return x11
        return x1
        # return x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11
        # return x1 + x2
        # return x1
        # return x1 + x2 + x3


class SPELL_bak(torch.nn.Module):
    def __init__(self, channels, feature_dim=1024, dropout=0, dropout_a=0, da_true=False):
        self.channels = channels
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.dropout_a = dropout_a
        self.da_true = da_true
        super(SPELL, self).__init__()
        self.feat_dim = 128
        self.layer_first_face = nn.Linear(128, 2)
        self.layer_first_background = nn.Linear(128, 2)
        self.layer011 = nn.Linear(512, self.channels[0])
        self.layer011_128 = nn.Linear(128, self.channels[0])
        self.layer011_128_2 = nn.Linear(128, self.channels[0])
        self.layer011_128_3 = nn.Linear(128, self.channels[0])
        self.layer011_128_4 = nn.Linear(128, self.channels[0])
        self.layer011_128_5 = nn.Linear(256, self.channels[0])
        self.layer_att = nn.Linear(128, 1)
        self.att_weight = nn.Linear(4, 4)
        
        self.x1_res = nn.Linear(128, 2)
        self.x2_res = nn.Linear(128, 2)
        self.x3_res = nn.Linear(64, 2)
        
        self.batch01 = BatchNorm(self.channels[0])

        self.layer11 = EdgeConv(nn.Sequential(nn.Linear(2*self.channels[0], self.channels[0]), nn.ReLU(), nn.Linear(self.channels[0], self.channels[0])))
        self.batch11 = BatchNorm(self.channels[0])
        self.layer12 = EdgeConv(nn.Sequential(nn.Linear(2*self.channels[0], self.channels[0]), nn.ReLU(), nn.Linear(self.channels[0], self.channels[0])))
        self.batch12 = BatchNorm(self.channels[0])
        self.layer13 = EdgeConv(nn.Sequential(nn.Linear(2*self.channels[0], self.channels[0]), nn.ReLU(), nn.Linear(self.channels[0], self.channels[0])))
        self.batch13 = BatchNorm(self.channels[0])

        self.layer23 = SAGEConv(self.channels[0], self.channels[1])
        # self.layer23 = EdgeConv(nn.Sequential(nn.Linear(2*self.channels[0], self.channels[0]), nn.ReLU(), nn.Linear(self.channels[0], self.channels[1])))
        self.layer22 = SAGEConv(self.channels[0], self.channels[1])
        self.batch21 = BatchNorm(self.channels[1])

        self.layer31 = SAGEConv(self.channels[1], 2)
        self.layer32 = SAGEConv(self.channels[1], 2)
        self.layer33 = SAGEConv(self.channels[1], 2)
        self.layer34 = SAGEConv(self.channels[0], 2)
        
        self.layernew1 = EdgeConv(nn.Sequential(nn.Linear(2*self.channels[1], self.channels[1]), nn.ReLU(), nn.Linear(self.channels[1], 2)))
        self.layernew2 = EdgeConv(nn.Sequential(nn.Linear(2*self.channels[1], self.channels[1]), nn.ReLU(), nn.Linear(self.channels[1], 2)))
        self.fc1 = nn.Linear(self.channels[1], 2)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x_face_av = x[:,:self.feat_dim]
        x_face_body_av = x[:,self.feat_dim:2*self.feat_dim]
        x_face_av1 = x[:,3*self.feat_dim:4*self.feat_dim]
        x_face_av2 = x[:,4*self.feat_dim:5*self.feat_dim]
        x_face_av3 = x[:,5*self.feat_dim:6*self.feat_dim]
        x_face_av4 = x[:,6*self.feat_dim:7*self.feat_dim]
        x_face_av5 = x[:,7*self.feat_dim:8*self.feat_dim]
        x_face_av6 = x[:,8*self.feat_dim:9*self.feat_dim]
        x_face_av7 = x[:,9*self.feat_dim:10*self.feat_dim]
        x_face_av8 = x[:,10*self.feat_dim:11*self.feat_dim]
        x_face_av9 = x[:,11*self.feat_dim:12*self.feat_dim]
        x_face_av10 = x[:,12*self.feat_dim:13*self.feat_dim]
        x_face_av11 = x[:,13*self.feat_dim:14*self.feat_dim]

        x_face_av = F.dropout(x_face_av, p=0.6, training=self.training)
        x_face_body_av = F.dropout(x_face_body_av, p=0.6, training=self.training)
        x_face_av1 = F.dropout(x_face_av1, p=0.6, training=self.training)
        x_face_av2 = F.dropout(x_face_av2, p=0.6, training=self.training)
        x_face_av3 = F.dropout(x_face_av3, p=0.6, training=self.training)
        x_face_av4 = F.dropout(x_face_av4, p=0.6, training=self.training)
        x_face_av5 = F.dropout(x_face_av5, p=0.6, training=self.training)
        x_face_av6 = F.dropout(x_face_av6, p=0.6, training=self.training)
        x_face_av7 = F.dropout(x_face_av7, p=0.6, training=self.training)
        x_face_av8 = F.dropout(x_face_av8, p=0.6, training=self.training)
        x_face_av9 = F.dropout(x_face_av9, p=0.6, training=self.training)
        x_face_av10 = F.dropout(x_face_av10, p=0.6, training=self.training)
        x_face_av11 = F.dropout(x_face_av11, p=0.6, training=self.training)
    
        w0 = self.layer_att(x_face_av)
        w1 = self.layer_att(x_face_av1)
        w2 = self.layer_att(x_face_av2)
        w3 = self.layer_att(x_face_av3)
        w4 = self.layer_att(x_face_av4)
        w5 = self.layer_att(x_face_av5)
        w6 = self.layer_att(x_face_av6)
        w7 = self.layer_att(x_face_av7)
        w8 = self.layer_att(x_face_av8)
        w9 = self.layer_att(x_face_av9)
        w10 = self.layer_att(x_face_av10)
        w11 = self.layer_att(x_face_av11)
        w12 = self.layer_att(x_face_body_av)
        
        w = torch.concat([w0,w6,w10,w12],dim=1)
        w = self.att_weight(w)    
        
        # x = self.layer011_128(w[:,0].unsqueeze(1)*x_face_av
        #                       +w[:,1].unsqueeze(1)*x_face_av6
        #                       +w[:,2].unsqueeze(1)*x_face_av10
        #                       +w[:,3].unsqueeze(1)*x_face_body_av)

        x = self.layer011_128(x_face_av + x_face_body_av)


        # x_face_av = x[:,:self.feature_dim//4]
        # x_background_av = x[:,self.feature_dim//4:self.feature_dim//2]
        # x_body_av = x[:,self.feature_dim//2:3*self.feature_dim//4]
        # x_face_body_av = x[:,3*self.feature_dim//4:]

        # x_face_av = F.dropout(x_face_av, p=0.6, training=self.training)
        # x_background_av = F.dropout(x_background_av, p=0.6, training=self.training)
        # x_body_av = F.dropout(x_body_av, p=0.6, training=self.training)
        # x_face_body_av = F.dropout(x_face_body_av, p=0.6, training=self.training)


        # x = self.layer011_128(x_face_av+x_face_body_av+x_background_av)

        
        # x = F.elu(x)
        # edge_index1 = edge_index[:, (edge_attr >= 0) & (edge_attr <= 0.9)]
        # edge_index2 = edge_index[:, (edge_attr <= 0) & (edge_attr >= -0.9)]
        # edge_index3 = edge_index[:, abs(edge_attr) <= 0.9]

        # edge_index1m, _ = dropout_adj(edge_index=edge_index1, p=self.dropout_a, training=self.training)
        # x1 = self.layer11(x, edge_index1m)
        # # x1 = self.batch11(x1)
        # x1 = F.elu(x1)
        # x1 = F.dropout(x1, p=self.dropout, training=self.training)
        # x1 = self.layer21(x1, edge_index1)
        # # x1 = self.batch21(x1)
        # x1 = F.elu(x1)
        # x1 = F.dropout(x1, p=self.dropout, training=self.training)

        # edge_index2m, _ = dropout_adj(edge_index=edge_index2, p=self.dropout_a, training=self.training)
        # x2 = self.layer12(x, edge_index2m)
        # # x2 = self.batch12(x2)
        # x2 = F.elu(x2)
        # x2 = F.dropout(x2, p=self.dropout, training=self.training)
        # x2 = self.layer21(x2, edge_index2)
        # # x2 = self.batch21(x2)
        # x2 = F.elu(x2)
        # x2 = F.dropout(x2, p=self.dropout, training=self.training)

        # # Undirected graph
        # edge_index3m, _ = dropout_adj(edge_index=edge_index3, p=self.dropout_a, training=self.training)
        # x3 = self.layer13(x, edge_index3m)
        # # x3 = self.batch13(x3)
        # x3 = F.elu(x3)
        # x3 = F.dropout(x3, p=self.dropout, training=self.training)
        # x3 = self.layer21(x3, edge_index3)
        # # x3 = self.batch21(x3)
        # x3 = F.elu(x3)
        # x3 = F.dropout(x3, p=self.dropout, training=self.training)

        # x1 = self.layer31(x1, edge_index1)
        # x2 = self.layer32(x2, edge_index2)
        # x3 = self.layer33(x3, edge_index3)

        # x = x1 + x2 + x3
        # return x



        # edge_index4 = edge_index[:, (edge_attr == 111) | (abs(edge_attr) > 1)]
        # edge_index4 = edge_index[:, (edge_attr == 111) | (abs(edge_attr) == 0)]  #94.94
        # edge_index4 = edge_index[:, (edge_attr == 111) | (abs(edge_attr) <= 0.04)] # 94.97
        edge_index4 = edge_index[:, (edge_attr == 111) | (abs(edge_attr) <= 0.05)] #95.0115
        # edge_index4 = edge_index[:, (edge_attr == 111) | ((edge_attr <= 0.05)&(edge_attr >= 0)) ] # 95.0087
        # edge_index4 = edge_index[:, (edge_attr == 111) | ((edge_attr >= -0.05)&(edge_attr <= 0)) ] # 95.0087
        # edge_index4 = edge_index[:, (edge_attr == 111) | (edge_attr == 0) ] # 95.0011
        # edge_index4 = edge_index[:, (edge_attr == 111) | (abs(edge_attr) <= 0.09)] # 94.97
        # edge_index4 = edge_index[:, (edge_attr == 111) | (abs(edge_attr) <= 0.15)] # 94.96
        # edge_index4 = edge_index[:, (edge_attr == 111) | (abs(edge_attr) <= 0.02)] # 
        # edge_index4 = edge_index[:, (abs(edge_attr) <= 0.05) & (abs(edge_attr) != 0)]
        # edge_index4 = edge_index[:, (edge_attr == 111) | (abs(edge_attr)<=0.06)]
        
        edge_index4m, _ = dropout_adj(edge_index=edge_index4, p=self.dropout_a, training=self.training)
        x4 = self.layer13(x, edge_index4m)
        x4 = F.elu(x4)
        x4 = F.dropout(x4, p=0.2, training=self.training)

        x4 = self.layer23(x, edge_index4)
        x4 = F.elu(x4)
        x4 = F.dropout(x4, p=0.2, training=self.training)

        x4 = self.layernew1(x4, edge_index4)

        return x4

        # edge_index5 = edge_index[:, edge_attr == 111]
        
        # edge_index5m, _ = dropout_adj(edge_index=edge_index5, p=self.dropout_a, training=self.training)
        # x5 = self.layer13(x, edge_index5m)
        # x5 = F.elu(x5)
        # x5 = F.dropout(x5, p=0.2, training=self.training)
        # x5 = self.layer23(x, edge_index5)
        # x5 = F.elu(x5)
        # x5 = F.dropout(x5, p=0.2, training=self.training)
        # # x5 = self.layer32(x5, edge_index5)
        # # x5 = self.fc1(x5)
        # x5 = self.layernew1(x5, edge_index5)



        
        
        
        edge_index1 = edge_index[:, (edge_attr >= 0) & (edge_attr <= 1)]
        edge_index2 = edge_index[:, (edge_attr <= 0) & (edge_attr >= -1)]
        edge_index3 = edge_index[:, abs(edge_attr) <= 1]
        edge_index5 = edge_index[:, edge_attr == 0]
        
        x_face = x[:,:self.feature_dim//2]
        x_background = x[:,self.feature_dim//2:]
        x_face_av = x_face[:,:128]
        x_face_v = x_face[:,128:256]
        x_face_a = x_face[:,256:]
        x_background_av = x_background[:,:128]
        x_background_v = x_background[:,128:256]
        x_background_a = x_background[:,256:]
        
        x_face_av = F.dropout(x_face_av, p=0.5, training=self.training)
        x_background_av = F.dropout(x_background_av, p=0.5, training=self.training)
        
        # x4 = self.layer_first_face(x_face_av)
        # x5 = self.layer_first_background(x_background_av)
        
        # x = self.layer011(torch.cat((x_face,x_background),dim=1))
        # x = self.layer011(x_face_v+x_face_a)
        # x = self.layer011(torch.cat((x_face_av,x_face_v+x_face_a,x_background_av, x_background_v+x_background_a),dim=1))
        # x = self.layer011(torch.cat((x_face_av,x_face_v,x_background_av, x_background_v),dim=1)) # 94.55
        # x = self.layer011_256(torch.cat((x_face_av, x_background_av),dim=1)) # 94.59
        x = torch.cat((self.layer011_128(x_face_av), self.layer011_128_2(x_background_av)),dim=1)
        # x = self.layer011_128(x_face_av) + self.layer011_128_2(x_background_av)
        # x = self.layer011_128(x_face_av) 
        # x = x_face_av + x_background_av
        
        # weight = self.layer011_weight(torch.cat((x_face_av, x_background_av),dim=1)) # 94.59
        # # weight = F.softmax(weight, dim=1)
        # weight = F.relu(weight)
        # x = x_face_av * weight[:,0].unsqueeze(1) + x_background_av * weight[:,1].unsqueeze(1)
        # x = self.layer011_128(x) 
        # x = self.layer011_128(x_face_v+x_face_a)  # 93.53
        # x = self.layer011_256(torch.cat((x_face_v+x_face_a, x_background_v+x_background_a),dim=1)) #93.92
        # x = self.layer011_256(torch.cat((x_face_v+x_face_a, x_background_av),dim=1)) # 93.83
        # x = self.layer011_128(x_background_v+x_background_a) # 80.52
        # x = self.batch01(x)
        x = F.elu(x)

        edge_index4m, _ = dropout_adj(edge_index=edge_index4, p=self.dropout_a, training=self.training)
        x4 = self.layer13(x, edge_index4m)
        x4 = F.elu(x4)
        x4 = F.dropout(x4, p=self.dropout, training=self.training)
        x4 = self.layer21(x4, edge_index4)
        x4 = F.elu(x4)
        x4 = F.dropout(x4, p=self.dropout, training=self.training)
        x4 = self.layer33(x4, edge_index4)
        return x4
    
        # edge_index1m, _ = dropout_adj(edge_index=edge_index1, p=self.dropout_a, training=self.training)
        # x1 = self.layer11(x, edge_index1m)
        # # x1 = self.batch11(x1)
        # x1 = F.relu(x1)
        # x1 = F.dropout(x1, p=self.dropout, training=self.training)
        # x1 = self.layer21(x1, edge_index1)
        # # x1 = self.batch21(x1)
        # x1 = F.relu(x1)
        # x1 = F.dropout(x1, p=self.dropout, training=self.training)

        # edge_index2m, _ = dropout_adj(edge_index=edge_index2, p=self.dropout_a, training=self.training)
        # x2 = self.layer12(x, edge_index2m)
        # # x2 = self.batch12(x2)
        # x2 = F.relu(x2)
        # x2 = F.dropout(x2, p=self.dropout, training=self.training)
        # x2 = self.layer21(x2, edge_index2)
        # # x2 = self.batch21(x2)
        # x2 = F.relu(x2)
        # x2 = F.dropout(x2, p=self.dropout, training=self.training)

        # # Undirected graph
        # edge_index3m, _ = dropout_adj(edge_index=edge_index3, p=self.dropout_a, training=self.training)
        # x3 = self.layer13(x, edge_index3m)
        # # x3 = self.batch13(x3)
        # x3 = F.relu(x3)
        # x3 = F.dropout(x3, p=self.dropout, training=self.training)
        # x3 = self.layer21(x3, edge_index3)
        # # x3 = self.batch21(x3)
        # x3 = F.relu(x3)
        # x3 = F.dropout(x3, p=self.dropout, training=self.training)

        # x1 = self.layer31(x1, edge_index1)
        # x2 = self.layer32(x2, edge_index2)
        # x3 = self.layer33(x3, edge_index3)

        # x = x1 + x2 + x3
        # # x = torch.sigmoid(x)

        return x