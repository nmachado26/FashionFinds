from src.dataset import DeepFashionCAPDataset
from src.const import base_path
import pandas as pd
import torch
import torch.utils.data
from src import const
from src.utils import parse_args_and_merge_const
from tensorboardX import SummaryWriter
import os

#extra imports
from torchvision import models
import torch.nn as nn
from src.base_networks import CustomUnetGenerator, ModuleWithAttr, VGG16Extractor
from torch.nn import functional as F


#super DIRTY but putting model in here to not mess w file path
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(SimpleModel, self).__init__()
        
        self.vgg16_extractor = VGG16Extractor()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        #batch_size = x.size(0)
        batch_size, channel_num, image_h, image_w = x['image'].size()

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x['image'], hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


if __name__ == '__main__':
    parse_args_and_merge_const()
    if os.path.exists('models') is False:
        os.makedirs('models')

    #used this repo's data preprocessing
    df = pd.read_csv(base_path + const.USE_CSV)
    train_df = df[df['evaluation_status'] == 'train']
    train_dataset = DeepFashionCAPDataset(train_df, mode=const.DATASET_PROC_METHOD_TRAIN)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=const.BATCH_SIZE, shuffle=True, num_workers=4)
    val_df = df[df['evaluation_status'] == 'test']
    val_dataset = DeepFashionCAPDataset(val_df, mode=const.DATASET_PROC_METHOD_VAL)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=const.VAL_BATCH_SIZE, shuffle=False, num_workers=4)
    val_step = len(val_dataloader)

    print("shape train_df:", train_df.shape)
    
    #iterate through dataloader once
    #trainiter = iter(dataloaders['train'])
    #features, labels = next(trainiter)
    #print("shapes: ", features.shape, labels.shape)
    

    #change model type here!!
    #model_name = 'VGG16'
    #print("1: ", const.USE_NET())

  
    #model = const.USE_NET()
    #model = model_name()
    vgg16 = False
    simple = True
    resNet = False

    model = models.vgg16(pretrained=True) #for safety of not declaring to None

    if vgg16:
        model = models.vgg16(pretrained=True)

        # Freeze model weights
        for param in model.parameters():
            param.requires_grad = False
   
        #add our own layers to transfer
        n_inputs = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
                          nn.Linear(n_inputs, 256), 
                          nn.ReLU(), 
                          nn.Dropout(0.4),
                          nn.Linear(256, n_classes),                   
                          nn.LogSoftmax(dim=1))
    elif simple:
        # Instantiate the model with hyperparameters
        dict_size = len(train_dataloader)
        model = SimpleModel(input_size=dict_size, output_size=dict_size, hidden_dim=12, n_layers=1)
    else: #resnet for now
        model = models.resnet50(pretrained=False)

        #for param in model.parameters():
            #param.requires_grad = False

        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))
    


    #print our param counts
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    #choose between gpu or local (uncomment one from below)
    #model = model.to('cuda')
    model = model.to(const.device)

    #if we are on various gpu's, can do them in parallel (uncomment if so)
    #model = nn.DataParallel(model)

    learning_rate = const.LEARNING_RATE
    #choosing loss function
    lossFn = nn.NLLLoss() #negative likelihood
    lossFn = nn.CrossEntropyLoss()
    #choosing optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(const.TRAIN_DIR)

    total_step = len(train_dataloader)
    step = 0
    n_epochs = 3
    for epoch in range(n_epochs):
        model.train()
        for i, sample in enumerate(train_dataloader):
            step += 1
            print("sample: ", sample)
            for key in sample:
                sample[key] = sample[key].to(const.device)
            output = model(sample)
            loss = loss_fn(output, sample)
            loss['all'].backward()
            optimizer.step()        
     
            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, const.NUM_EPOCH, i + 1, total_step, loss['all'].item()))
          
        learning_rate *= const.LEARNING_RATE_DECAY
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

