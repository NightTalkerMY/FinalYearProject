import torch
import torch.nn as nn

from model.Classifier import BGRU
from model.Encoder import visual_encoder, visual_encoder_no_t, audio_encoder
from model.Graph import GraphASD

class ASD_Model(nn.Module):
    def __init__(self, encoder_struct=None):
        super(ASD_Model, self).__init__()
        self.visualEncoder  = visual_encoder(encoder_struct)
        audio_encoder_struct = [1] + encoder_struct[1:]
        self.audioEncoder  = audio_encoder(audio_encoder_struct)
        self.GRU = BGRU(encoder_struct[-1])
        self.dim = encoder_struct[-1]
        
    def forward_visual_frontend(self, x):
        if len(x.shape) < 5:
            B, T, W, H = x.shape
              
            x = x.view(B, 1, T, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        # x = x / 255
        x = self.visualEncoder(x)
        return x

    def forward_audio_frontend(self, x):    
        x = x.unsqueeze(1).transpose(2, 3)     
        x = self.audioEncoder(x)
        return x

    def forward_audio_visual_backend(self, x1, x2):  
        x = x1 + x2 
        # x = x1
        x = self.GRU(x)   
        x = torch.reshape(x, (-1, self.dim))
        return x    

    def forward_visual_backend(self,x):
        x = torch.reshape(x, (-1, self.dim))
        return x

    def forward(self, audioFeature, visualFeature):
        audioEmbed = self.forward_audio_frontend(audioFeature)
        visualEmbed = self.forward_visual_frontend(visualFeature)
        outsAV = self.forward_audio_visual_backend(audioEmbed, visualEmbed)  
        outsV = self.forward_visual_backend(visualEmbed)
        outsA = self.forward_visual_backend(audioEmbed)

        return outsAV, outsV, outsA


class FrontASD_Model(nn.Module):
    def __init__(self):
        super(FrontASD_Model, self).__init__()
        
        self.visualEncoder  = visual_encoder()
        self.audioEncoder  = audio_encoder()
        self.graph = GraphASD([64,128],feature_dim=128, dropout=0.2, dropout_a=0)
        # self.GRU = BGRU(128)

    def forward_visual_frontend(self, x):
        B, T, W, H = x.shape  
        x = x.view(B, 1, T, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        x = self.visualEncoder(x)
        return x

    def forward_audio_frontend(self, x):    
        x = x.unsqueeze(1).transpose(2, 3)     
        x = self.audioEncoder(x)
        return x

    def forward_visual_backend(self,x):
        x = torch.reshape(x, (-1, 128))
        return x

    def forward_audio_visual_backend(self, x1, x2):  
        x = x1 + x2 
        x = self.GRU(x)   
        x = torch.reshape(x, (-1, 128))
        return x  
         
    def forward(self, audioFeature, visualFeature, edge_index, edge_attr, speakers=2):
        # indices = torch.arange(0, visualFeature.size(0), speakers).cuda() 
        # visualFeature = torch.index_select(visualFeature, 0, indices)
        audioEmbed = self.forward_audio_frontend(audioFeature)
        visualEmbed = self.forward_visual_frontend(visualFeature)
        indices = torch.arange(0, visualEmbed.size(0), speakers).cuda() 
        mainVisualEmbed = torch.index_select(visualEmbed, 0, indices)
        
        # visual classification
        outsV = self.forward_visual_backend(mainVisualEmbed)
        # audio-visual classification
        outsAV = self.graph(visualEmbed, audioEmbed, edge_index, edge_attr, speakers)  
        # outsAV = self.forward_audio_visual_backend(audioEmbed, visualEmbed)  

        return outsAV, outsV




class ASD_face_audio_Model(nn.Module):
    def __init__(self):
        super(ASD_face_audio_Model, self).__init__()
        
        self.visualEncoder  = visual_encoder_no_t()
        self.audioEncoder  = audio_encoder()
        self.GRU = BGRU(128)

    def forward_visual_frontend(self, x):
        B, T, W, H = x.shape  
        x = x.view(B, 1, T, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        # x = x / 255
        x = self.visualEncoder(x)
        return x

    def forward_audio_frontend(self, x):    
        x = x.unsqueeze(1).transpose(2, 3)     
        x = self.audioEncoder(x)
        return x

    def forward_audio_visual_backend(self, x1, x2):  
        x = x1 + x2 
        # x = x1
        x = self.GRU(x)   
        x = torch.reshape(x, (-1, 128))
        return x    

    def forward_visual_backend(self,x):
        x = torch.reshape(x, (-1, 128))
        return x

    def forward(self, audioFeature, visualFeature):
        # audioFeature:     batch x len x 13
        # visualFeature:    batch x 1 x 112 x 112
        audioEmbed = self.forward_audio_frontend(audioFeature)      # batch x len/4 x 128
        visualEmbed = self.forward_visual_frontend(visualFeature)   # batch x 1 x 128
        outsAV = self.forward_audio_visual_backend(audioEmbed, visualEmbed)  
        # outsV = self.forward_visual_backend(visualEmbed)

        return outsAV, visualEmbed.squeeze(1), audioEmbed

