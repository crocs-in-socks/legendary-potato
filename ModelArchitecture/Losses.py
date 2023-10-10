import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

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

class ContrastiveLoss(nn.Module):
    def __init__(self,temperature=0.05,batch_size=32,n_views=2,device='cuda:0'):
        super(ContrastiveLoss,self).__init__()
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




