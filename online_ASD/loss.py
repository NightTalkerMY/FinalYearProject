import torch
import torch.nn as nn
import torch.nn.functional as F

class lossAV(nn.Module):
	def __init__(self,encoder_struct=None):
		super(lossAV, self).__init__()
		self.criterion = nn.BCELoss()
		self.FC        = nn.Linear(encoder_struct[-1], 2)
		
	def forward(self, x, labels = None, r = 1):	
		x = x.squeeze(1)
		x = self.FC(x)
		if labels == None:
			predScore = x[:,1]
			predScore = predScore.t()
			predScore = predScore.view(-1).detach().cpu().numpy()
			return predScore
		else:
			x1 = x / r
			x1 = F.softmax(x1, dim = -1)[:,1]
			nloss = self.criterion(x1, labels.float())
			predScore = F.softmax(x, dim = -1)
			predLabel = torch.round(F.softmax(x, dim = -1))[:,1]
			correctNum = (predLabel == labels).sum().float()
			return nloss, predScore, predLabel, correctNum


class lossV(nn.Module):
	def __init__(self, encoder_struct=None):
		super(lossV, self).__init__()
		self.criterion = nn.BCELoss()
		self.FC        = nn.Linear(encoder_struct[-1], 2)

	def forward(self, x, labels, r = 1):	
		x = x.squeeze(1)
		x = self.FC(x)
		
		x = x / r
		x = F.softmax(x, dim = -1)

		nloss = self.criterion(x[:,1], labels.float())
		return nloss


class npair_loss(nn.Module):
    def __init__(self):
        super(npair_loss, self).__init__()
        self.w = nn.Parameter(torch.FloatTensor([10]))
        self.b = nn.Parameter(torch.FloatTensor([-5]))
        
    def forward(self, audio_emb, visual_emb):
        batch_size = audio_emb.size(0)
        device = audio_emb.device
        torch.clamp(self.w, 1e-6)
        target = torch.eye(batch_size).to(device)
        # calculate AV and VA
        logit_av = torch.matmul(audio_emb, torch.transpose(visual_emb, 0, 1))
        logit_av = self.w * logit_av + self.b
        l_av = torch.mean(torch.sum(- target * F.log_softmax(logit_av, -1), -1))        
        l_va = torch.mean(torch.sum(- target * F.log_softmax(logit_av, 0), -1))   
        # calculate AA
        logit_aa = torch.matmul(audio_emb, torch.transpose(audio_emb, 0, 1))
        logit_aa = self.w * logit_aa + self.b
        diag_aa = torch.diag(torch.diag(logit_aa))
        diag_av = torch.diag(torch.diag(logit_av))
        logit_aa = logit_aa - diag_aa + diag_av
        l_aa = torch.mean(torch.sum(- target * F.log_softmax(logit_aa, -1), -1))   
        # calculate VV
        logit_vv = torch.matmul(visual_emb, torch.transpose(visual_emb, 0, 1))
        logit_vv = self.w * logit_vv + self.b
        diag_vv = torch.diag(torch.diag(logit_vv))
        diag_av = torch.diag(torch.diag(logit_av))
        logit_vv = logit_vv - diag_vv + diag_av
        l_vv = torch.mean(torch.sum(- target * F.log_softmax(logit_vv, -1), -1))   
        
        # loss = l_av + l_va + 0.2*(l_aa + l_vv)
        # loss = l_aa + l_vv
        loss = l_av + l_va
        
        return loss

class lossContrast(nn.Module):
	def __init__(self):
		super(lossContrast, self).__init__()
		self.npair_loss = npair_loss()

	def forward(self, predScore, outsV, outsA):	
		# predScore:	batch*len x 2
		# outsV:		batch x 128
		# outsA:		batch x len x 128
		batch_size, seq_len, dim = outsA.shape
		predScore = predScore.reshape((batch_size, seq_len, 2))		# batch x len x 2
		postive_val = predScore[:,:,1].unsqueeze(2)		# batch x len x 1

		
		emb_a_postive = torch.mean(outsA*postive_val, dim=1)		# batch x 128
		emb_v = outsV * torch.max(postive_val, dim=1)[0]			# batch x 128

		emb_a_postive = emb_a_postive / torch.norm(emb_a_postive, dim=1, keepdim=True)	# batch x 128
		emb_v = emb_v / torch.norm(emb_v, dim=1, keepdim=True)							# batch x 128

		loss = self.npair_loss(emb_a_postive, emb_v)

		return loss