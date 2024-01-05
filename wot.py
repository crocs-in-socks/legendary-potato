import torch
import torch.nn.functional as F

from tqdm import tqdm

conf_consistency_criterion = torch.nn.KLDivLoss(size_average=False, reduce=False).cuda()
count = 0
for i in tqdm(range(100000)):
    count += 1

    output = torch.randn(1, 60, 8, 8, 8).cuda()
    flip_op = torch.randn(1, 60, 8, 8, 8).cuda()   
    # CONS_LOSS
    output += 1e-7
    flip_op += 1e-7
    cons_loss_a = conf_consistency_criterion(F.logsigmoid(output), F.sigmoid(flip_op)).sum()
    print(cons_loss_a)
    if cons_loss_a < 0:
        break

print(count)
print('done')