import torch
import torch.nn as nn


    

class Audio_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Audio_Block, self).__init__()

        self.relu = nn.ReLU()

        self.m_3 = nn.Conv2d(in_channels, out_channels, kernel_size = (3, 1), padding = (1, 0), bias = False)
        self.bn_m_3 = nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001)
        self.t_3 = nn.Conv2d(out_channels, out_channels, kernel_size = (1, 3), padding = (0, 1), bias = False)
        self.bn_t_3 = nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001)
        
        self.m_5 = nn.Conv2d(in_channels, out_channels, kernel_size = (5, 1), padding = (2, 0), bias = False)
        self.bn_m_5 = nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001)
        self.t_5 = nn.Conv2d(out_channels, out_channels, kernel_size = (1, 5), padding = (0, 2), bias = False)
        self.bn_t_5 = nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001)
        
        self.last = nn.Conv2d(out_channels, out_channels, kernel_size = (1, 1), padding = (0, 0), bias = False)
        self.bn_last = nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001)

    def forward(self, x):

        x_3 = self.relu(self.bn_m_3(self.m_3(x)))
        x_3 = self.relu(self.bn_t_3(self.t_3(x_3)))

        x_5 = self.relu(self.bn_m_5(self.m_5(x)))
        x_5 = self.relu(self.bn_t_5(self.t_5(x_5)))

        x = x_3 + x_5
        x = self.relu(self.bn_last(self.last(x)))

        # x_3 = self.relu(self.m_3(x))
        # x_3 = self.relu(self.t_3(x_3))

        # x_5 = self.relu(self.m_5(x))
        # x_5 = self.relu(self.t_5(x_5))

        # x = x_3 + x_5
        # x = self.relu(self.last(x))

        return x


class Visual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, is_down = False):
        super(Visual_Block, self).__init__()

        self.relu = nn.ReLU()

        if is_down:
            self.s_3 = nn.Conv3d(in_channels, out_channels, kernel_size = (1, 3, 3), stride = (1, 2, 2), padding = (0, 1, 1), bias = False)
            self.bn_s_3 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)
            self.t_3 = nn.Conv3d(out_channels, out_channels, kernel_size = (3, 1, 1), padding = (1, 0, 0), bias = False)
            self.bn_t_3 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)

            self.s_5 = nn.Conv3d(in_channels, out_channels, kernel_size = (1, 5, 5), stride = (1, 2, 2), padding = (0, 2, 2), bias = False)
            self.bn_s_5 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)
            self.t_5 = nn.Conv3d(out_channels, out_channels, kernel_size = (5, 1, 1), padding = (2, 0, 0), bias = False)
            self.bn_t_5 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)
        else:
            self.s_3 = nn.Conv3d(in_channels, out_channels, kernel_size = (1, 3, 3), padding = (0, 1, 1), bias = False)
            self.bn_s_3 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)
            self.t_3 = nn.Conv3d(out_channels, out_channels, kernel_size = (3, 1, 1), padding = (1, 0, 0), bias = False)
            self.bn_t_3 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)

            self.s_5 = nn.Conv3d(in_channels, out_channels, kernel_size = (1, 5, 5), padding = (0, 2, 2), bias = False)
            self.bn_s_5 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)
            self.t_5 = nn.Conv3d(out_channels, out_channels, kernel_size = (5, 1, 1), padding = (2, 0, 0), bias = False)
            self.bn_t_5 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)

        self.last = nn.Conv3d(out_channels, out_channels, kernel_size = (1, 1, 1), padding = (0, 0, 0), bias = False)
        self.bn_last = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)

    def forward(self, x):

        x_3 = self.relu(self.bn_s_3(self.s_3(x)))
        x_3 = self.relu(self.bn_t_3(self.t_3(x_3)))

        x_5 = self.relu(self.bn_s_5(self.s_5(x)))
        x_5 = self.relu(self.bn_t_5(self.t_5(x_5)))

        x = x_3 + x_5

        x = self.relu(self.bn_last(self.last(x)))

        # x_3 = self.relu(self.s_3(x))
        # x_3 = self.relu(self.t_3(x_3))

        # x_5 = self.relu(self.s_5(x))
        # x_5 = self.relu(self.t_5(x_5))

        # x = x_3 + x_5

        # x = self.relu(self.last(x))


        return x

class Visual_Block_no_t(nn.Module):
    def __init__(self, in_channels, out_channels, is_down = False):
        super(Visual_Block_no_t, self).__init__()

        self.relu = nn.ReLU()

        if is_down:
            self.s_3 = nn.Conv3d(in_channels, out_channels, kernel_size = (1, 3, 3), stride = (1, 2, 2), padding = (0, 1, 1), bias = False)
            self.bn_s_3 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)

            self.s_5 = nn.Conv3d(in_channels, out_channels, kernel_size = (1, 5, 5), stride = (1, 2, 2), padding = (0, 2, 2), bias = False)
            self.bn_s_5 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)
        else:
            self.s_3 = nn.Conv3d(in_channels, out_channels, kernel_size = (1, 3, 3), padding = (0, 1, 1), bias = False)
            self.bn_s_3 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)

            self.s_5 = nn.Conv3d(in_channels, out_channels, kernel_size = (1, 5, 5), padding = (0, 2, 2), bias = False)
            self.bn_s_5 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)

        self.last = nn.Conv3d(out_channels, out_channels, kernel_size = (1, 1, 1), padding = (0, 0, 0), bias = False)
        self.bn_last = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)

    def forward(self, x):

        x_3 = self.relu(self.bn_s_3(self.s_3(x)))

        x_5 = self.relu(self.bn_s_5(self.s_5(x)))

        x = x_3 + x_5

        x = self.relu(self.bn_last(self.last(x)))

        # x_3 = self.relu(self.s_3(x))
        # x_3 = self.relu(self.t_3(x_3))

        # x_5 = self.relu(self.s_5(x))
        # x_5 = self.relu(self.t_5(x_5))

        # x = x_3 + x_5

        # x = self.relu(self.last(x))


        return x


class visual_encoder(nn.Module):
    def __init__(self, encoder_struct=None):
        super(visual_encoder, self).__init__()

        self.block1 = Visual_Block(encoder_struct[0], encoder_struct[1], is_down = True)
        # self.pool1 = nn.MaxPool3d(kernel_size = (1, 3, 3), stride = (1, 2, 2), padding = (0, 1, 1))
        self.pool1_2D = nn.MaxPool2d(kernel_size = (3, 3), stride = (2, 2), padding = (1, 1))
        self.block2 = Visual_Block(encoder_struct[1], encoder_struct[2])
        # self.pool2 = nn.MaxPool3d(kernel_size = (1, 3, 3), stride = (1, 2, 2), padding = (0, 1, 1))
        self.pool2_2D = nn.MaxPool2d(kernel_size = (3, 3), stride = (2, 2), padding = (1, 1))
        
        self.block3 = Visual_Block(encoder_struct[2], encoder_struct[3])

        # self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.__init_weight()
        # for name, param in self.named_parameters():
        #     if name == 'block3.s_3.weight':
        #         print(f"{name}: {param.data}")
        # input()

    def forward(self, x):

        x = self.block1(x)
        b1, b2 = x.shape[1], x.shape[2]
        x = x.reshape(x.shape[0], b1*b2, x.shape[-2], x.shape[-1])
        x = self.pool1_2D(x)
        x = x.reshape(x.shape[0], b1, b2, x.shape[-2], x.shape[-1])
        
        x = self.block2(x)
        b1, b2 = x.shape[1], x.shape[2]
        x = x.reshape(x.shape[0], b1*b2, x.shape[-2], x.shape[-1])
        x = self.pool2_2D(x)
        x = x.reshape(x.shape[0], b1, b2, x.shape[-2], x.shape[-1])

        x = self.block3(x)

        x = x.transpose(1,2)
        B, T, C, W, H = x.shape  
        x = x.reshape(B, T, C, W*H)
        x = torch.mean(x, dim = 3)
        
        # x = self.maxpool(x)

        # x = x.view(B, T, C)  
        
        return x

    def __init_weight(self):

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class visual_encoder_no_t(nn.Module):
    def __init__(self):
        super(visual_encoder_no_t, self).__init__()

        self.block1 = Visual_Block_no_t(1, 32, is_down = True)
        # self.pool1 = nn.MaxPool3d(kernel_size = (1, 3, 3), stride = (1, 2, 2), padding = (0, 1, 1))
        self.pool1_2D = nn.MaxPool2d(kernel_size = (3, 3), stride = (2, 2), padding = (1, 1))
        self.block2 = Visual_Block_no_t(32, 64)
        # self.pool2 = nn.MaxPool3d(kernel_size = (1, 3, 3), stride = (1, 2, 2), padding = (0, 1, 1))
        self.pool2_2D = nn.MaxPool2d(kernel_size = (3, 3), stride = (2, 2), padding = (1, 1))
        
        self.block3 = Visual_Block_no_t(64, 128)

        # self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.__init_weight()
        # for name, param in self.named_parameters():
        #     if name == 'block3.s_3.weight':
        #         print(f"{name}: {param.data}")
        # input()

    def forward(self, x):

        x = self.block1(x)
        b1, b2 = x.shape[1], x.shape[2]
        x = x.reshape(x.shape[0], b1*b2, x.shape[-2], x.shape[-1])
        x = self.pool1_2D(x)
        x = x.reshape(x.shape[0], b1, b2, x.shape[-2], x.shape[-1])
        
        x = self.block2(x)
        b1, b2 = x.shape[1], x.shape[2]
        x = x.reshape(x.shape[0], b1*b2, x.shape[-2], x.shape[-1])
        x = self.pool2_2D(x)
        x = x.reshape(x.shape[0], b1, b2, x.shape[-2], x.shape[-1])

        x = self.block3(x)

        x = x.transpose(1,2)
        B, T, C, W, H = x.shape  
        x = x.reshape(B, T, C, W*H)
        x = torch.mean(x, dim = 3)
        
        # x = self.maxpool(x)

        # x = x.view(B, T, C)  
        
        return x

    def __init_weight(self):

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class audio_encoder(nn.Module):
    def __init__(self, encoder_struct=None):
        super(audio_encoder, self).__init__()
        
        self.block1 = Audio_Block(encoder_struct[0], encoder_struct[1])
        # self.pool1 = nn.MaxPool3d(kernel_size = (1, 1, 3), stride = (1, 1, 2), padding = (0, 0, 1))
        self.pool1_1D = nn.MaxPool1d(kernel_size = 3, stride = 2, padding = 1)
        
        self.block2 = Audio_Block(encoder_struct[1], encoder_struct[2])
        # self.pool2 = nn.MaxPool3d(kernel_size = (1, 1, 3), stride = (1, 1, 2), padding = (0, 0, 1))
        self.pool2_1D = nn.MaxPool1d(kernel_size = 3, stride = 2, padding = 1)
        
        self.block3 = Audio_Block(encoder_struct[2], encoder_struct[3])

        self.__init_weight()
            
    def forward(self, x):
        x = self.block1(x)
        b1 = x.shape[0]
        b2 = x.shape[1]
        x = x.reshape(b1*b2, x.shape[2], x.shape[3])
        x = self.pool1_1D(x)
        x = x.reshape(b1,b2,x.shape[1], x.shape[2])
        
        x = self.block2(x)
        b1 = x.shape[0]
        b2 = x.shape[1]
        x = x.reshape(b1*b2, x.shape[2], x.shape[3])
        x = self.pool2_1D(x)
        x = x.reshape(b1,b2,x.shape[1], x.shape[2])

        x = self.block3(x)

        x = torch.mean(x, dim = 2, keepdim = True)
        x = x.squeeze(2).transpose(1, 2)
        
        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()