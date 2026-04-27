import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, SAGEConv, EdgeConv
from torch_geometric.utils.dropout import dropout_adj
from torch_geometric.utils import add_self_loops, remove_self_loops, to_dense_adj
from torch_geometric.utils import dropout_adj, sort_edge_index
from model.fusion import *
import ipdb


class MSSG_bak(torch.nn.Module):
    # TODO:Large face processing...
    def __init__(self):
        super(MSSG, self).__init__()
        self.feat_dim = 128
        
        self.max_speaker = 3
        self.speaker_fc = nn.Linear(self.max_speaker, 16)
        
        self.spatial_fc = nn.Linear(4, 16)
        
        self.layer_frontend1 = nn.Linear(128, 64)
        self.layer_frontend2 = nn.Linear(128+32, 64)
         
        self.layer1 = EdgeConv(nn.Sequential(nn.Linear(2*64, 2)))
        self.layer2 = SAGEConv(64, 2)
        print(" Graph Model para number = %.2f M"%(sum(param.numel() for param in self.parameters()) / 1000 / 1000))            
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        face_size = x[:,-6].unsqueeze(1)
        mask_index = face_size[:, 0] >= 80
        mask = torch.zeros(x.shape[0], 7).cuda()
        mask[mask_index] = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]).cuda()
        mask[~mask_index] = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]).cuda()
        mask = mask.unsqueeze(2)
        
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
                        # face_right_feat.unsqueeze(1),
                        x_face_down_av.unsqueeze(1),
                        ),dim=1)        # batch x num x feat     
        x = F.dropout(x, p=0.9, training=self.training)
        x = x * mask
        x = torch.sum(x, dim=1)
        
        x1 = self.layer_frontend1(x)
        x2 = self.layer_frontend2(torch.cat((x, speaker_num_feat, spatial_feat), dim=1))                               # batch x 64        
        x = x1 + x2
        
        edge_index1, _ = dropout_adj(edge_index=edge_index1, p=0.01, training=self.training)
        x1 = self.layer1(x, edge_index1)       
        x2 = self.layer2(x, edge_index1)
        
        return x1 + x2


class MSSG_Flops(torch.nn.Module):
    def __init__(self):
        super(MSSG, self).__init__()
        self.feat_dim = 128
        
        self.max_speaker = 3
        self.speaker_fc = nn.Linear(self.max_speaker, 16)
        
        self.spatial_fc = nn.Linear(4, 16)
        
        self.layer_frontend1 = nn.Linear(128, 64)
        self.layer_frontend2 = nn.Linear(128+32, 64)
         
        self.layer1 = EdgeConv(nn.Sequential(nn.Linear(2*64, 2)))
        self.layer2 = SAGEConv(64, 2)
        print(" Graph Model para number = %.2f M"%(sum(param.numel() for param in self.parameters()) / 1000 / 1000))            
    def forward(self, x, edge_index, edge_attr):

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

class MSSG_bak(torch.nn.Module):
    def __init__(self):
        super(MSSG, self).__init__()
        self.feat_dim = 128
        
        self.max_speaker = 3
        self.speaker_fc = nn.Linear(self.max_speaker, 16)
        
        self.spatial_fc = nn.Linear(4, 16)
        
        self.layer_frontend1 = nn.Linear(128, 64)
        self.layer_frontend2 = nn.Linear(128+32, 64)
         
        self.layer1 = EdgeConv(nn.Sequential(nn.Linear(2*64, 2)))
        self.layer2 = SAGEConv(64, 2)
        
        print(" Graph Model para number = %.2f M"%(sum(param.numel() for param in self.parameters()) / 1000 / 1000))            
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        edge_index1 = edge_index[:, (edge_attr == 111) | (edge_attr == 0)] # co-speaker in one frame 

        x_face_av =             x[:,:self.feat_dim]
        x_face_body_av =        x[:,1*self.feat_dim:2*self.feat_dim]
        x_face_large_av =       x[:,2*self.feat_dim:3*self.feat_dim]
        x_background_av =       x[:,3*self.feat_dim:4*self.feat_dim]
        x_face_small_av =       x[:,4*self.feat_dim:5*self.feat_dim]
        # face_right_feat =       x[:,5*self.feat_dim:6*self.feat_dim]
        x_face_down_av =        x[:,5*self.feat_dim:6*self.feat_dim]
        x_face_body_large_av =  x[:,6*self.feat_dim:7*self.feat_dim]

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
        
        # return x2
        
        return x1 + x2

class MSSG(torch.nn.Module):
    def __init__(self):
        super(MSSG, self).__init__()
        self.feat_dim = 128
        
        self.max_speaker = 3# 2,3,4,5,6,7,8,9,10
        self.speaker_fc = nn.Linear(self.max_speaker, 16)
        
        self.spatial_fc = nn.Linear(4, 16)
        
        self.layer_frontend1 = nn.Linear(128, 64)
        self.layer_frontend2 = nn.Linear(128+32, 64)
        
        # self.layer_frontend2_1 = nn.Linear(128+16, 64)
        # self.layer_frontend2_2 = nn.Linear(128+16, 64)
         
        self.layer1 = EdgeConv(nn.Sequential(nn.Linear(2*64, 2)))
        self.layer2 = SAGEConv(64, 2)
        
        print(" Graph Model para number = %.2f M"%(sum(param.numel() for param in self.parameters()) / 1000 / 1000))            
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        edge_index1 = edge_index[:, (edge_attr == 111) | (edge_attr == 0)] # co-speaker in one frame 

        x_face_av =             x[:,:self.feat_dim]
        x_face_body_av =        x[:,1*self.feat_dim:2*self.feat_dim]
        x_face_large_av =       x[:,2*self.feat_dim:3*self.feat_dim]
        x_background_av =       x[:,3*self.feat_dim:4*self.feat_dim]
        x_face_small_av =       x[:,4*self.feat_dim:5*self.feat_dim]
        # face_right_feat =       x[:,5*self.feat_dim:6*self.feat_dim]
        x_face_down_av =        x[:,5*self.feat_dim:6*self.feat_dim]
        x_face_body_large_av =  x[:,6*self.feat_dim:7*self.feat_dim]

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
                        x_face_down_av.unsqueeze(1),
                        ),dim=1)        # batch x num x feat     
        x = F.dropout(x, p=0.9, training=self.training)
        x = torch.sum(x, dim=1) / 3
        
        x1 = self.layer_frontend1(x)
        x2 = self.layer_frontend2(torch.cat((x, speaker_num_feat, spatial_feat), dim=1))                               # batch x 64        
        # x2 = self.layer_frontend2_1(torch.cat((x, speaker_num_feat), dim=1))                               # batch x 64        
        # x2 = self.layer_frontend2_2(torch.cat((x, spatial_feat), dim=1))                               # batch x 64        
        x = x1 + x2
        
        edge_index1, _ = dropout_adj(edge_index=edge_index1, p=0.01, training=self.training)
        x1 = self.layer1(x, edge_index1)       
        x2 = self.layer2(x, edge_index1)
        
        # return x2
        
        return x1 + x2


class MSSG_best(torch.nn.Module):
    def __init__(self):
        super(MSSG, self).__init__()
        self.feat_dim = 128
        
        self.max_speaker = 3
        self.speaker_fc = nn.Linear(self.max_speaker, 16)
        
        self.spatial_fc = nn.Linear(4, 16)
        
        self.layer_frontend1 = nn.Linear(128, 64)
        self.layer_frontend2 = nn.Linear(128+32, 64)
         
        self.layer1 = EdgeConv(nn.Sequential(nn.Linear(2*64, 2)))
        self.layer2 = SAGEConv(64, 2)
        print(" Graph Model para number = %.2f M"%(sum(param.numel() for param in self.parameters()) / 1000 / 1000))            
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
                        # face_right_feat.unsqueeze(1),
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