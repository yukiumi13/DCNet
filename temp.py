import numpy as np
import torch
a = torch.load('bmvc_cv.pth',map_location = torch.device('cpu'))
b = torch.load('bmvc_cv_single.pth',map_location = torch.device('cpu'))