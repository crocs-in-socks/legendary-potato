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

class Frequency_loss(nn.Module):
    def __init__(self,weight):
        super().__init__()
        self.weights = weight
    def forward(self,x,y):
        return torch.sum(self.weights*torch.abs(x-y))


class MS_SSIMLoss(nn.Module):
    def __init__(self,data_range = 1.0,win_size=5,win_sigma=1.5,channel=1,spatial_dims=3):
        super(MS_SSIMLoss,self).__init__()
        self.eps = 1e-7
        self.data_range = data_range
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.channel = channel
        self.spatial_dims = spatial_dims
        self.ms_ssim = MS_SSIM(data_range = self.data_range,win_size = self.win_size,win_sigma = self.win_sigma,channel = self.channel,spatial_dims =self.spatial_dims)
    def forward(self,x,y):
        return 1-self.ms_ssim(x,y)

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
    
class LogCoshDiceLoss(nn.Module):
    def __init__(self,weight=None):
        super().__init__()

    def forward(self, x, target, wt = 1):
        smooth = 1
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(x * target, dims)
        cardinality = torch.sum(x + target, dims)

        dice_score = ((2. * intersection + smooth) / (cardinality + smooth))
        dice_loss = (1 - dice_score)
        
        return torch.log((torch.exp(dice_loss) + torch.exp(-dice_loss)) / 2.0).mean()

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
    def __init__(self, temperature=0.07, device='cpu', num_voxels=10500):
        super().__init__()
        self.temperature = temperature
        self.struct_element = np.ones((5, 5, 5), dtype=bool)
        self.device = device
        self.max_pixels = num_voxels
        self.coefficient = 1
    
    def forward(self, Zs, pixel_mask, subtracted_mask=None, brain_mask=None):

        number_of_features = Zs.shape[1]
        positive_mask = (pixel_mask[:, 1] == 1).squeeze(0)

        if subtracted_mask is not None:
            negative_mask = (subtracted_mask == 1).squeeze(0, 1)
        elif brain_mask is not None:
            negative_mask = torch.logical_and(brain_mask[:, 0] == 1, pixel_mask[:, 0] == 1).squeeze(0)
        else:
            negative_mask = (pixel_mask[:, 0] == 1).squeeze(0)

        positive_pixels = Zs[:, :, positive_mask].permute(0, 2, 1).reshape(-1, number_of_features)
        negative_pixels = Zs[:, :,negative_mask].permute(0, 2, 1).reshape(-1, number_of_features)

        if positive_pixels.shape[0] > self.max_pixels:
            random_indices = torch.randint(0, positive_pixels.size(0), size=(self.max_pixels,))
            positive_pixels = positive_pixels[random_indices]
        
        if positive_pixels.shape[0] < negative_pixels.shape[0]:
            random_indices = torch.randint(0, negative_pixels.size(0), size=(positive_pixels.shape[0],))
            negative_pixels = negative_pixels[random_indices]
        elif negative_pixels.shape[0] > self.max_pixels:
            random_indices = torch.randint(0, negative_pixels.size(0), size=(self.max_pixels,))
            negative_pixels = negative_pixels[random_indices]

        pixels = torch.cat([positive_pixels, negative_pixels])
        labels = torch.tensor([1] * positive_pixels.shape[0] + [0] * negative_pixels.shape[0]).to(self.device)

        pixels = F.normalize(pixels)
        dot = torch.matmul(pixels, pixels.T)
        dot = torch.div(dot, self.temperature)
        dot = F.normalize(dot)
        exp = torch.exp(dot)

        class_mask = torch.eq(labels, labels.unsqueeze(1))
        class_mask[torch.arange(len(labels)), torch.arange(len(labels))] = False

        positive_mask = exp * class_mask
        positive_mask[positive_mask == 0] = 1
        # positive_mask = exp * class_mask + (~class_mask).float()
        negative_mask = exp * (~class_mask)

        denominator = torch.sum(negative_mask, dim=1) - torch.diagonal(exp)
        full_term = torch.log(positive_mask) - torch.log(denominator)
        loss = -(1 / len(labels)) * torch.sum(full_term) * self.coefficient
        
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

def determine_multiclass_accuracy(pred, target):
    pred_vect = (pred > 0.5).float()
    target_vect = (target > 0.5).float()

    correct_cases = (pred_vect == target_vect)
    true_pos = torch.sum(correct_cases)

    accuracy = true_pos/(pred_vect.shape[0] * pred_vect.shape[1])
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
    
    # print()
    # print(true_positives)
    # print(true_negatives)
    # print(false_positives)
    # print(false_negatives)
    # print()

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

from ModelArchitecture.Losses_unified_focal import AsymmetricFocalTverskyLoss

class DICELoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self,pred,target,wt=1):
        #print(wt.shape)
        dice = AsymmetricFocalTverskyLoss(delta=0.5,gamma=0)(pred,target)
        return dice

class FocalDICELoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self,pred,target,wt=1):
        #print(wt.shape)
        dice = AsymmetricFocalTverskyLoss(delta=0.5,gamma=0.5)(pred,target)
        return dice
    
class WBCE_DICELoss(nn.Module):
    def __init__(self,):
        super().__init__()
        pass
    def forward(self,pred,target,wt=1):
        #print(wt.shape)
        dice = DiceLoss()(pred[:,1],target[:,1])
        wbce = BCE_Loss_Weighted(weight=5)(pred,target,wt)
        return dice + wbce

class WBCE_FOCALDICELoss(nn.Module):
    def __init__(self,):
        super().__init__()

        pass
    def forward(self,pred,target,wt=1):
        dice = AsymmetricFocalTverskyLoss(delta=0.5)(pred,target) # this becomes dice
        wbce = BCE_Loss_Weighted(weight=5)(pred,target,wt)
        return dice + wbce

class WBCE_DICELoss(nn.Module):
    def __init__(self,):
        super().__init__()
        pass
    def forward(self,pred,target,wt=1):
        #print(wt.shape)
        dice = DiceLoss()(pred[:,1],target[:,1])
        wbce = BCE_Loss_Weighted(weight=5)(pred,target,wt)
        return dice + wbce

class WBCE_FOCALDICELoss(nn.Module):
    def __init__(self,):
        super().__init__()

        pass
    def forward(self,pred,target,wt=1):
        dice = AsymmetricFocalTverskyLoss(delta=0.5)(pred,target) # this becomes dice
        wbce = BCE_Loss_Weighted(weight=5)(pred,target,wt)
        return dice + wbce
    
#################################################################################################
# Losses Taken from https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
#################################################################################################
ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1,wt=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss
    
#PyTorch
ALPHA = 0.5
BETA = 0.5

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky
    
ALPHA = 0.5
BETA = 0.5
GAMMA = 1

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky

ALPHA = 0.5 # < 0.5 penalises FP more, > 0.5 penalises FN more
CE_RATIO = 0.5 #weighted contribution of modified CE loss compared to Dice loss

class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, eps=1e-9):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()    
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        inputs = torch.clamp(inputs, eps, 1.0 - eps)       
        out = - (ALPHA * ((targets * torch.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)
        
        return combo