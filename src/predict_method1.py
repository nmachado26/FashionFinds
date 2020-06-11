from src.dataset import DeepFashionCAPDataset
from src.const import base_path
import pandas as pd
import torch
import torch.utils.data
from src import const
from src.utils import parse_args_and_merge_const
from tensorboardX import SummaryWriter
import os

from torchvision import datasets, models, transforms
import torch.nn as nn
from src.base_networks import CustomUnetGenerator, ModuleWithAttr, VGG16Extractor
from torch.nn import functional as F
from sklearn.model_selection import train_test_split

import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage import io, transform
from random import randrange
import torch.nn.functional as F
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torchvision.transforms as transforms
from torch.autograd import Variable

class multi_output_model(torch.nn.Module):
    def __init__(self, model_core,dd):
        super(multi_output_model, self).__init__()        
        self.resnet_model = model_core
        self.x1 =  nn.Linear(512,256)
        nn.init.xavier_normal_(self.x1.weight)
        self.bn1 = nn.BatchNorm1d(256,eps = 1e-2)
        #heads
        self.y1o = nn.Linear(256,2)            
        nn.init.xavier_normal_(self.y1o.weight)
        self.y2o = nn.Linear(256,1000)
        nn.init.xavier_normal_(self.y2o.weight)
        
    def forward(self, x):
        x = self.resnet_model(x)     
        x1 = self.bn1(F.relu(self.x1(x)))
        # heads
        y1o = F.softmax(self.y1o(x1),dim=1)
        y2o = torch.sigmoid(self.y2o(x1)) #should be sigmoid
        return y1o, y2o

def make_list_from_file(file_path):
    with open(file_path) as fp:
        line = fp.readline()
        ret = []
        firstLine = fp.readline()
        for line in fp:
            item = ret.append(line.split(' ')[0])   
    return ret

def image_loader(image_path):
    """load image, returns cuda tensor"""
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = loader(image).float()
    #print(image.shape)
    image = Variable(image, requires_grad=False)
    image = image.unsqueeze(0)  
    return image.cuda()  #assumes using GPU

def extract_label(label_list, pred_array,top_n=1):
    pred_max = torch.topk(pred_array,top_n)[1]
    out_list = []
    for i in pred_max[0]:
        out_list.append(label_list[i])
    return out_list

if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dd = .2
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 512)

    model1 = multi_output_model(model_ft,dd)
    model1 = model1.to(device)
    model1.load_state_dict(torch.load('models/multiresnet2Epoch.pkl'))
    model1.eval()
    
    imsize = 224
    loader = transforms.Compose([transforms.Resize((imsize,imsize)),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    dir_path = '/home/ubuntu/lowResData/test/'
    i = 1
    directory = os.fsencode(dir_path)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith('.jpg' or '.jpeg'):
            image_path = dir_path + filename
            image = image_loader(image_path)
            y_pred = model1(image)
            category_name = make_list_from_file('/home/ubuntu/lowResData/Anno/list_category_cloth.txt')
            attributes = make_list_from_file('/home/ubuntu/lowResData/Anno/list_attr_cloth.txt')	
            print('Image ', i)
            print('Image path: ', image_path)
            print('Category name: ', extract_label(category_name,y_pred[0]))
            print('Attributes: ', extract_label(attributes,y_pred[1], 5))
            i += 1
            continue
        else:
            continue

