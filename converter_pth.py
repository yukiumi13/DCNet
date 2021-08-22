####################################
# 去掉并行模型的参数文件中的Module.前缀 #
####################################
import torch, torchvision
from collections import OrderedDict
dict = torch.load('bmvc_cv.pth',map_location=torch.device('cpu'))
new_dict = OrderedDict()
for k, v in dict.items():
    name = k[7:]
    new_dict[name] = v
torch.save(new_dict,'bmvc_cv_single.pth')