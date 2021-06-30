import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLossWithCosine(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLossWithCosine, self).__init__()
        self.margin = margin
        self.cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-8)
    
    def forward(self, output1, output2, label):
        cos_similarity_score = self.cos_similarity(output1 , output2)
        cos_distance = 1 - cos_similarity_score
        #pos = (label) * torch.pow(cos_distance, 2)
        pos = (label) * cos_distance
        #neg = (1- label) * torch.pow(torch.clamp(self.margin - cos_distance, min=0.0), 2)
        neg = (1- label) * torch.clamp(self.margin - cos_distance, min=0.0)
        loss_contrastive = torch.mean( pos + neg )
        return loss_contrastive