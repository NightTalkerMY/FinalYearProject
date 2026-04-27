import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, time, numpy, os, subprocess, pandas
from subprocess import PIPE

from loss import lossAV, lossV, lossContrast
from model.Model import ASD_Model, ASD_face_audio_Model
from tqdm import tqdm
import ipdb
import random
import string
import time
from sklearn.metrics import average_precision_score
import copy 

def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    random.seed(int(time.time() * 1e6) % 1e6)
    return ''.join(random.choice(characters) for i in range(length))

def list_sorted_files(directory):
    files = os.listdir(directory)
    # files = [f for f in files if os.path.isfile(os.path.join(directory, f))]
    files.sort(reverse=True)
    return files

class ASD(nn.Module):
    def __init__(self, lr = 0.001, encoder_struct=None, lrDecay = 0.95, **kwargs):
        super(ASD, self).__init__()
        if len(encoder_struct) < 4:
            encoder_struct = [1] + encoder_struct        
        self.model = ASD_Model(encoder_struct).cuda()
        self.lossAV = lossAV(encoder_struct).cuda()
        self.lossV = lossV(encoder_struct).cuda()
        
        
        self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=100)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1000 / 1000))

    def train_network(self, loader, epoch, type, **kwargs):
        self.train()
        index, top1, lossV, lossAV, loss = 0, 0, 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        r = 1.3 - 0.02 * (epoch - 1)
        for num, (audioFeature, visualFeature, labels) in enumerate(loader, start=1):
            self.zero_grad()

            outsAV, outsV, _ = self.model(audioFeature[0].cuda(), visualFeature[0].cuda())
            
            labels = labels[0].reshape((-1)).cuda() # Loss
            # if type == 'background':
            #     labels = 1 - labels
            nlossAV, _, _, prec = self.lossAV(outsAV, labels, r)
            nlossV = self.lossV(outsV, labels, r)
            nloss = nlossAV + 0.5 * nlossV
            
            lossV += nlossV.detach().cpu().numpy()
            lossAV += nlossAV.detach().cpu().numpy()
            loss += nloss.detach().cpu().numpy()
            top1 += prec

            nloss.backward()
            self.optim.step()

            index += len(labels)
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] r: %2f, Lr: %5f, Training: %.2f%%, "    %(epoch, r, lr, 100 * (num / loader.__len__())) + \
            " LossV: %.5f, LossAV: %.5f, Loss: %.5f, ACC: %2.2f%% \r"  %(lossV/(num), lossAV/(num), loss/(num), 100 * (top1/index)))
            sys.stderr.flush()
            # break

        self.scheduler.step()
        sys.stdout.write("\n")      

        return loss/num, lr

    def evaluate_network(self, loader, evalCsvSave, evalOrig, **kwargs):
        self.eval()
        predScores = []
        target_total = []
        soft_total = []
        iii = 0
        for audioFeature, visualFeature, labels in tqdm(loader):
            iii += 1
            with torch.no_grad():                
                outsAV, outsV, _ = self.model(audioFeature[0].cuda(), visualFeature[0].cuda())
                labels = labels[0].reshape((-1)).cuda()    
                # if type == 'background':
                #     labels = 1 - labels         
                _, predScore, _, _ = self.lossAV(outsAV, labels)    
                # predScore = predScore[:,1].detach().cpu().numpy()
                scores = predScore[:, 1].tolist()
                soft_total.extend(scores)
                target_total.extend(labels[:].tolist())
                # break
        
        mAP = average_precision_score(target_total, soft_total) * 100
        # evalLines = open(evalOrig).read().splitlines()[1:] 
        # labels = []
        # labels = pandas.Series( ['SPEAKING_AUDIBLE' for line in evalLines])
        # scores = pandas.Series(predScores)
        # evalRes = pandas.read_csv(evalOrig)
        # evalRes = evalRes[0:scores.shape[0]]
        # evalRes['score'] = scores
        # evalRes['label'] = 1 - labels
        # evalRes.drop(['label_id'], axis=1,inplace=True)
        # evalRes.drop(['instance_id'], axis=1,inplace=True)
        # eval_res_file = os.path.join('./', 'res_'+ generate_random_string(10)+'.csv')
        # evalRes.to_csv(eval_res_file, index=False)
        # evalOri = pandas.read_csv(evalOrig)
        # evalOri = evalOri[0:scores.shape[0]]
        # tmp_file = os.path.join('./', 'tmp_'+ generate_random_string(10)+'.csv')
        # evalOri.to_csv(tmp_file, index=False)
        # cmd = "python -O utils/get_ava_active_speaker_performance.py -g %s -p %s "%(tmp_file, eval_res_file)
        # mAP = float(str(subprocess.run(cmd, shell=True, stdout=PIPE, stderr=PIPE).stdout).split(' ')[2][:5])
        # try:
        #     os.remove(tmp_file)
        #     os.remove(eval_res_file)
        # except OSError as e:
        #     print("Delete wrong file!")
        return mAP

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path, weights_only=True)
        for name, param in loadedState.items():
            origName = name;
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)
    
    def loadParameters_multi(self, path):
        selfState = self.state_dict()
        avg_state_dict = copy.deepcopy(selfState)
        for key in avg_state_dict:
            avg_state_dict[key] = torch.zeros_like(avg_state_dict[key])
        checkpoint_paths = list_sorted_files(path)[:7]
        print(checkpoint_paths)
        print(f'Average {len(checkpoint_paths)} models')
        for checkpoint in checkpoint_paths:
            # model = torch.load(os.path.join(path, checkpoint), map_location=torch.device('cpu'))
            model = torch.load(os.path.join(path, checkpoint), weights_only=True)
            for key in avg_state_dict:
                avg_state_dict[key] += model[key]
        
        for key in avg_state_dict:
            type(avg_state_dict[key])
            avg_state_dict[key] = avg_state_dict[key] / len(checkpoint_paths)
        
        for name, param in avg_state_dict.items():
            origName = name
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != avg_state_dict[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), avg_state_dict[origName].size()))
                continue
            selfState[name].copy_(param)




class ASD_face_audio(nn.Module):
    def __init__(self, lr = 0.001, lrDecay = 0.95, **kwargs):
        super(ASD_face_audio, self).__init__()        
        self.model = ASD_face_audio_Model().cuda()
        self.lossAV = lossAV().cuda()
        self.lossContrast = lossContrast().cuda()
        self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=100)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1000 / 1000))

    def train_network(self, loader, epoch, type, **kwargs):
        self.train()
        index, top1, lossContrast, lossAV, loss = 0, 0, 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        # r = 1.3 - 0.02 * (epoch - 1)
        r = 1
        for num, (audioFeature, visualFeature, labels) in enumerate(loader, start=1):
            # audioFeature:     1 x batch x len x 13
            # visualFeature:    1 x batch x 1 x 112 x 112
            # labels:           1 x batch x len
            self.zero_grad()

            # outsAV:   batch*len x 128
            # outsV:    batch x 128
            # outsA:    batch x len x 128
            outsAV, outsV, outsA = self.model(audioFeature[0].cuda(), visualFeature[0].cuda())
            
            labels = labels[0].reshape((-1)).cuda() # Loss
            nlossAV, predScore, _, prec = self.lossAV(outsAV, labels, r)
            nlossContrast = self.lossContrast(predScore, outsV, outsA)
            nloss = nlossAV + 0.0001 * nlossContrast
            
            lossContrast += nlossContrast.detach().cpu().numpy()
            lossAV += nlossAV.detach().cpu().numpy()
            loss += nloss.detach().cpu().numpy()
            top1 += prec

            nloss.backward()
            self.optim.step()

            index += len(labels)
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] r: %2f, Lr: %5f, Training: %.2f%%, "    %(epoch, r, lr, 100 * (num / loader.__len__())) + \
            " LossV: %.5f, LossAV: %.5f, Loss: %.5f, ACC: %2.2f%% \r"  %(lossContrast/(num), lossAV/(num), loss/(num), 100 * (top1/index)))
            sys.stderr.flush()

        self.scheduler.step()
        sys.stdout.write("\n")      

        return loss/num, lr

    def evaluate_network(self, loader, evalCsvSave, evalOrig, **kwargs):
        self.eval()
        predScores = []
        target_total = []
        soft_total = []
        iii = 0
        for audioFeature, visualFeature, labels in tqdm(loader):
            iii += 1
            with torch.no_grad():                
                outsAV, outsV, outsA = self.model(audioFeature[0].cuda(), visualFeature[0].cuda())
                labels = labels[0].reshape((-1)).cuda()    
       
                _, predScore, _, _ = self.lossAV(outsAV, labels)    
                scores = predScore[:, 1].tolist()
                soft_total.extend(scores)
                target_total.extend(labels[:].tolist())
        
        mAP = average_precision_score(target_total, soft_total) * 100
        return mAP

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path, weights_only=True)
        for name, param in loadedState.items():
            origName = name;
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)