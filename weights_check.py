import torch
from auxiliary_functions import *
ssm = single_salicency_model(drop_rate = 0.2, layers=12)
ssm = torch.nn.DataParallel(ssm, device_ids=[0, 1, 2, 3])
ssm.cuda()
ssm.load_state_dict(torch.load('bmvc_current_distributed.pth'))
for name , para in ssm.named_parameters():
    print(name,'size=',para.size())
