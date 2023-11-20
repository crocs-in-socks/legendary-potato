import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import numpy as np
import matplotlib.pyplot as plt
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from skimage.morphology import binary_dilation

class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss,self).__init__()
        self.eps = 1e-7
    def forward(self,x,y):

        # c1 = (0.2)**2
        # c2 = (0.3)**2
        # c3 = c2/2
        #
        # mean_lum_x = torch.mean(x, dim=(1, 2, 3, 4),keepdim=True)
        # mean_lum_y = torch.mean(x, dim=(1, 2, 3, 4),keepdim=True)
        #
        #
        # l = 2*(mean_lum_y*mean_lum_x+c1)/(mean_lum_x**2 + mean_lum_y**2 + c1)
        #
        # std_x = np.sqrt(1/(64**3-1))*torch.sqrt(torch.sum((x-mean_lum_x),dim=(1,2,3,4)))
        # std_y = np.sqrt(1/(64**3-1))*torch.sqrt(torch.sum((x-mean_lum_y),dim=(1,2,3,4)))
        #
        # c = (2*std_x*std_y  + c2)/(std_x**2 +std_y**2 +c2)
        # std_xy = np.sqrt(1/(64**3-1))*torch.sum((x-mean_lum_y)*(y-mean_lum_y))
        #
        # s = (std_xy + c3)/(std_x*std_y + c3)
        # print(l,c,s)

        return 1-ms_ssim(x,y,data_range=1,win_size=5)

class BCE_Loss(nn.Module):
    def __init__(self):
        super(BCE_Loss, self).__init__()
        self.eps = 1e-7
    def forward(self,x,target):
        x = torch.sigmoid(x)
        return torch.mean((-target*torch.log(x+self.eps) - (1-target)*torch.log(1-x+self.eps)))
    
class BCE_Loss_Weighted(nn.Module):
    def __init__(self,weight=None):
        super(BCE_Loss_Weighted, self).__init__()
        self.eps = 1e-7
        self.weight = weight
    def forward(self,x,target,wt=1):
        self.weight = torch.tensor(wt)
        self.weight = torch.nan_to_num(self.weight,0,0,0)
        if(self.weight.sum()==0):
            self.weight =1 
        
        return torch.mean(self.weight*(-target*torch.log(x+self.eps) - (1-target)*torch.log(1-x+self.eps)))


class DiceLoss(nn.Module):
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()
        self.eps = 1e-7

    def forward(self, x, target,wt=1):
        #num_classes = target.shape[1]  # Channels first
        #target = target.type(x.type())
        
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(x * target, dims)
        cardinality = torch.sum(x + target, dims)

        dice_loss = ((2. * intersection + self.eps) / (cardinality + self.eps))
        return (1 - dice_loss.mean())

class VoxelwiseSupConMSELoss(nn.Module):

    def __init__(self, MSE_weight=0.5, SupCon_weight=1, temperature=0.07, device='cpu'):
        super().__init__()
        self.SupCon_weight = SupCon_weight
        self.MSE_weight = MSE_weight
        self.MSE_criterion = nn.MSELoss()
        self.SupCon_criterion = VoxelwiseSupConLoss_inImage(device=device)
    
    def forward(self, Zs, class_labels, subtracted, feature_diff, input_diff):
        SupCon_loss = self.SupCon_criterion(Zs, class_labels, subtracted)
        #print('SupConLoss calculated.')
        MSE_loss = self.MSE_criterion(feature_diff, input_diff)
        #print('MSELoss calculated.')

        loss = self.SupCon_weight*SupCon_loss + self.MSE_weight*MSE_loss
        return loss

class VoxelwiseSupConLoss_inImage(nn.Module):
    def __init__(self, temperature=0.07, device='cpu'):
        super().__init__()
        self.temperature = temperature
        self.struct_element = np.ones((5, 5, 5), dtype=bool)
        self.device = device
        self.max_pixels = 10500
    
    def forward(self, Zs, pixel_mask, subtracted_mask):

        number_of_features = Zs.shape[1]
        positive_mask = (pixel_mask[:, 1] == 1).squeeze(0)
        negative_mask = (subtracted_mask == 1).squeeze(0, 1)

        positive_pixels = Zs[:, :, positive_mask].permute(0, 2, 1).reshape(-1, number_of_features)
        negative_pixels = Zs[:, :,negative_mask].permute(0, 2, 1).reshape(-1, number_of_features)

        if positive_pixels.shape[0] > self.max_pixels:
            random_indices = torch.randperm(positive_pixels.size(0))[:self.max_pixels]
            positive_pixels = positive_pixels[random_indices]

        if negative_pixels.shape[0] > self.max_pixels:
            random_indices = torch.randperm(negative_pixels.size(0))[:self.max_pixels]
            negative_pixels = negative_pixels[random_indices]

        pixels = torch.cat([positive_pixels, negative_pixels])
        labels = torch.tensor([1] * positive_pixels.shape[0] + [0] * negative_pixels.shape[0]).to(self.device)

        pixels = F.normalize(pixels)
        # pixels = torch.matmul(pixels, pixels.T)
        # pixels = torch.div(pixels, self.temperature)
        # pixels = F.normalize(pixels)
        # exp = torch.exp(pixels)
        dot = torch.matmul(pixels, pixels.T)
        dot = torch.div(dot, self.temperature)
        dot = F.normalize(dot)
        exp = torch.exp(dot)

        class_mask = torch.eq(labels, labels.unsqueeze(1))
        class_mask[torch.arange(len(labels)), torch.arange(len(labels))] = False

        positive_mask = exp * class_mask
        positive_mask[positive_mask == 0] = 1
        negative_mask = exp * (~class_mask)

        denominator = torch.sum(negative_mask, dim=1) - torch.diagonal(exp)
        full_term = torch.log(positive_mask) - torch.log(denominator)
        loss = -(1 / len(labels)) * torch.sum(full_term)
        
        return loss

class SupervisedContrastiveLoss(nn.Module):
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, Zs, class_labels):

        number_of_samples = Zs.shape[0]
        # class_labels = 1 - torch.argmax(class_labels, dim=1)
        Zs = F.normalize(Zs)
        dot = torch.matmul(Zs, Zs.T)
        dot = torch.div(dot, self.temperature)
        # dot = F.normalize(dot)
        exp = torch.exp(dot)

        class_mask = torch.eq(class_labels, class_labels.unsqueeze(1))
        class_mask[torch.arange(number_of_samples), torch.arange(number_of_samples)] = False
        
        positive_mask = exp * class_mask
        positive_mask[positive_mask == 0] = 1
        negative_mask = exp * (~class_mask)

        denominator = torch.sum(negative_mask, dim=1) - torch.diagonal(exp)
        full_term = torch.log(positive_mask) - torch.log(denominator)
        loss = -(1 / (number_of_samples)) * torch.sum(full_term)

        # loss = 0
        # for row_idx in range(number_of_samples):
        #     row = exp[row_idx]
        #     # print(f'Positive row sum: {torch.sum(row[class_mask[row_idx]])}')
        #     # print(f'Negative row sum: {torch.sum(row[~class_mask[row_idx]])}')
        #     # print(f'Row diagonal: {row[row_idx]}')
        #     denominator = torch.sum(row[~class_mask[row_idx]]) - row[row_idx]
        #     temp = torch.log(row[class_mask[row_idx]]) - torch.log(denominator)
        #     temp = torch.sum(temp)
        #     temp = (-1 / (number_of_samples-1)) * temp
        #     loss += temp
        
        return loss
    
def determine_class_accuracy(pred, target):
    pred_vect = 1 - torch.argmax(pred, dim=1)
    target_vect = 1 - torch.argmax(target, dim=1)

    # for i in range(pred_vect.shape[0]):
    #     print(pred_vect[i], pred[i])
    # exit(0)

    correct_cases = (pred_vect == target_vect)
    true_pos = torch.sum(correct_cases)

    accuracy = true_pos/pred_vect.shape[0]
    return accuracy

def determine_accuracy_metrics(pred, target):
    pred_vect = 1- torch.argmax(pred, dim=1)
    target_vect = 1 - torch.argmax(target, dim=1)

    true_positives = torch.sum((pred_vect == 1) & (target_vect == 1))
    true_negatives = torch.sum((pred_vect == 0) & (target_vect == 0))
    false_positives = torch.sum((pred_vect == 1) & (target_vect == 0))
    false_negatives = torch.sum((pred_vect == 0) & (target_vect == 1))

    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)

    return recall, precision, true_positives, true_negatives, false_positives, false_negatives


class ContrastiveLoss(nn.Module):

    def __init__(self, temperature=0.05, batch_size=32, n_views=2, device='cuda:0'):
        super().__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.n_views = n_views
        self.device = device

    def forward(self,features):
        self.batch_size = features.shape[0]//2
        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels
    
class DiceLossWeighted(nn.Module):
    def __init__(self, weight=None):
        super(DiceLossWeighted, self).__init__()
        self.eps = 1e-7
        self.weight = weight

    def forward(self, x, target,wt = 1):
        #num_classes = target.shape[1]  # Channels first
        #target = target.type(x.type())
        self.weight = wt
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(x * target, dims)
        cardinality = torch.sum(x + target, dims)

        dice_loss = ((2. * intersection + self.eps) / (cardinality + self.eps))
        return (1 - (dice_loss*self.weight).mean())

def determine_dice_metric(pred, target):
    smooth = 1.

    n_classes = pred.shape[1]  # b*c*h*w
    avg_dice = 0.0
    for i in range(n_classes):
        pred_vect = pred[:, i, :, :].contiguous().view(-1)
        target_vect = target[:, i, :, :].contiguous().view(-1)
        intersection = (pred_vect * target_vect).sum()
        dice = (2. * intersection + smooth) / (torch.sum(pred_vect) + torch.sum(target_vect) + smooth)
        avg_dice += dice
    return avg_dice / n_classes
