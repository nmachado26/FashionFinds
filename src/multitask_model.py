
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
from torchvision import datasets, models, transforms
import torch.nn as nn
from src.base_networks import CustomUnetGenerator, ModuleWithAttr, VGG16Extractor
from torch.nn import functional as F
from sklearn.model_selection import train_test_split

#xtra extra lol
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


'''
 ret = {}
        ret['image'] = image
        ret['category_type'] = category_type
        ret['category_label'] = category_label
        ret['landmark_vis'] = landmark_vis
        ret['landmark_in_pic'] = landmark_in_pic
        ret['landmark_pos'] = landmark_pos
        ret['landmark_pos_normalized'] = landmark_pos_normalized
        ret['attr'] = attr

'''
if __name__ == '__main__':
    parse_args_and_merge_const()
    if os.path.exists('models') is False:
        os.makedirs('models')

    df = pd.read_csv(base_path + const.USE_CSV)
    train_df = df[df['evaluation_status'] == 'train']
    X = df['image_name']
    Y_list = list(df)
    Y_list.remove('image_name')
    #print("Y",Y_list)
    y = df[Y_list]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train = X_train.values.tolist()
    X_test = X_test.values.tolist()
    print("X_train = ", X_train[:4])
    #name_map = make_map_from_file()
    #train
    #all300kcategorystrings = y_train['category_name']
    #print("300k:",all300kcategorystrings)

#    for k in range(len(all300kcategorystrings)):
 #       zeros = [0 for i in range(50)]
        
    all_labels = y_train['category_label']
    print("all labels:", all_labels)
    
    
        
    #categories = ['Anorak', 'Blazer', 'Blouse', 'Bomber', 'Button-Down', 'Cardigan', 'Flannel', 'Halter', 'Henley', 'Hoodie', 'Jacket', 'Jersey', 'Parka', 'Peacoat', 'Poncho', 'Sweater', 'Tank', 'Tee', 'Top', 'Turtleneck', 'Capris', 'Chinos', 'Culottes', 'Cutoffs', 'Gauchos', 'Jeans', 'Jeggings', 'Jodhpurs', 'Joggers', 'Leggings', 'Sarong', 'Shorts', 'Skirt', 'Sweatpants', 'Sweatshorts', 'Trunks', 'Caftan', 'Cape', 'Coat', 'Coverup', 'Dress', 'Jumpsuit', 'Kaftan', 'Kimono', 'Nightdress', 'Onesie', 'Robe', 'Romper', 'Shirtdress', 'Sundress']

    all_labels = y_train[['category_label']]
    

    gender = ['category_label','category_name','category_type']
    gender_test = y_train[gender].values.tolist()[:5]
   
    print("y_train[categorylabel,name,type]>>>>>>>>>>", gender_test) 
    fin_list = []
    for i,val in enumerate(all_labels.values.tolist()):
        if i<5:
           print("val arr:",val)
        #print('val:',val)
        v = val[0]
        if i<5:
            print("v>>>", v)
        empty_arr = [0 for i in range(50)]
        num = v
        empty_arr[num] = 1
        fin_list.append(empty_arr)
    
    cat_nodes = 50
    cat_train = fin_list
    print("cat train: >>>>>>>>>",cat_train[:5])

    attributes = ['attr_%d'%i for i in range(1000)]
    attr_train = y_train[attributes]
    attr_nodes = attr_train.shape[1]
    attr_train = attr_train.values.tolist()

    #print("attr_train = ", attr_train)[:4]
    #didnt do test set
    #categories = ['category_label', 'category_type']


    all_labels_test = y_test[['category_label']]
    fin_list_test = []
    for val in all_labels_test.values.tolist():
        #print('val:',val)
        v = val[0]
        empty_arr = [0 for i in range(50)]
        num = v
        empty_arr[num] = 1
        fin_list_test.append(empty_arr)
    cat_nodes = 50
    cat_test = fin_list_test


    #cat_test = y_test[categories]
    #cat_nodes = cat_test.shape[1]
    #print("cat nodes:", cat_nodes)
    #print("cat_nodes.shape: ", cat_nodes.shape)
    #cat_test = cat_test.values.tolist()

    attributes = ['attr_%d'%i for i in range(1000)]
    #print("Attributes:", attributes)
    attr_test = y_test[attributes]
    attr_nodes = attr_test.shape[1]
    #print("attr_nodes.shape: ", attr_nodes.shape)
    attr_test = attr_test.values.tolist()

    




    class fgo_dataset(torch.utils.data.Dataset):
        def __init__(self,king_of_lists, transform=None,base_path=base_path):
            """
            Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            """
            self.king_of_lists = king_of_lists
            self.transform = transform
        
        def __getitem__(self,index):
        #random_index = randrange(len(self.king_of_lists[index]))
        
            imgname = self.king_of_lists[0][index]
            #img1 = io.imread(base_path + imgname)
            #img1 = img1.convert('RGB')
            image = Image.open(base_path + imgname)
            img1 = image.convert('RGB')

            cat = self.king_of_lists[1][index] # gender
            attr = self.king_of_lists[2][index] # region
            if self.transform is not None:
                img1 = self.transform(img1)
            list_of_labels = [torch.from_numpy(np.array(cat)),
                      torch.from_numpy(np.array(attr))]
        #list_of_labels = torch.FloatTensor(list_of_labels)

        #torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
        #print(list_of_labels)
            return img1, list_of_labels[0],list_of_labels[1]
    
        def __len__(self):
            return len(self.king_of_lists[0])    
    
    batch_size = 16
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224,224)),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_lists = [X_train, cat_train, attr_train]
    test_lists = [X_test, cat_test, attr_test]

    training_dataset = fgo_dataset(king_of_lists = train_lists,
                               transform = data_transforms['train'] )
    test_dataset = fgo_dataset(king_of_lists = test_lists,
                           transform = data_transforms['val'] )

    dataloaders_dict = {'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True),
                   'val':torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
                   }
    dataset_sizes = {'train':len(train_lists[0]),
                'val':len(test_lists[0])}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)    
    
    learning_rate = 0.001
    

    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()
        print("BEGIN TRAINING")

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 100

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                print("PHASE:", phase)
                if phase == 'train':
                    #scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                cat_corrects = 0.0
                attr_corrects= list()
                total_attrs = list()
                #print(">>>>>>>>attr correct type:", attr_corrects)
            
                # Iterate over data.
                for i, tup in enumerate(dataloaders_dict[phase]):
                    inputs, cat, attr = tup
                    inputs = inputs.to(device) 
                    #print("input size:", input.size())
                
                    cat = cat.to(device)
                    attr = attr.to(device)
                    #for testing
                    #if i == 100: break
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    if i == -1:#change to 1 if want to print
                        print("cat:",torch.max(cat.float(), 1))
                        print("Attr max shape:",attr.float().shape)
                        print("Attr shape:",torch.max(attr.float(), 1))
                        
                    #print(">>>>>>>>attr correct type2:", attr_corrects)
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        #if i == 1:
                            #print(">>>>>>>>>>>>>>>>>>>>> inputs:", inputs.size())
                        
                        outputs = model(inputs)
                        if i == -1:#change to 0 to print
                            print("cat:",torch.max(cat.float(), 1))
                            print("Attr  shape:",attr.float().shape)
                            print("Attr max:",torch.max(attr.float(), 1))
                            print("outputs[1]:",outputs[1].shape)
                        loss0 = criterion[0](outputs[0], torch.max(cat.float(), 1)[1])
                        #loss1 = criterion[1](outputs[1], torch.max(attr.float(), 1)[1]) #dont want to be doing this
                        loss1 = criterion[1](outputs[1], attr.float())
                        #print("loss1: {:.4f} loss2:{:.4f}".format(loss0, loss1))
                        # backward + optimize only if in training phase
                        if phase == 'train':
                        
                            loss = loss0 + loss1
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                            if (i+1) % 500 == 0:
                                perc = float(i)*100/len(dataloaders_dict[phase])
                                print('------------ {} {:.4f}% loss: {:.4f} loss cat: {:.4f} loss atr: {:.4f}'.format(phase,perc,loss,loss0,loss1))
                                #cat_corrects += torch.sum(torch.max(outputs[0], 1)[1] == torch.max(cat, 1)[1])
                                #attr_corrects += torch.sum(torch.max(outputs[1], 1)[1] == torch.max(attr, 1)[1])
                                #attr_corrects.append(float((np.rint(outputs[4].cpu().detach().numpy()) == attr.float)))
                                 #print('{} running loss: {:.4f}'.format(phase,running_loss))

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    cat_corrects += torch.sum(torch.max(outputs[0], 1)[1] == torch.max(cat, 1)[1])
                    attr_corrects.append(sum(np.rint(outputs[1].cpu().detach().numpy()) == attr.float().cpu().detach().numpy()).astype(float))
                    total_attrs.append(float((attr.size()[0]*attr.size(1))))
                
                print("ALMOST DONE EPOCH")
                epoch_loss = running_loss / dataset_sizes[phase]
                cat_acc = cat_corrects.double() / dataset_sizes[phase]
                attr_acc = np.sum(attr_corrects)/sum(total_attrs)
                #attr_acc = float(np.sum(attr_corrects))/np.sum(total_attrs)
                #attr_acc = cat_acc 
                print('{} total loss: {:.4f} cat loss: {:.4f} attr loss: {:.4f}'.format(phase,loss,loss0,loss1))
                print('{} cat_Acc: {:.4f} attr_acc: {:.4f}'.format(
                    phase, cat_acc,attr_acc))

                #learning_rate *= 0.9
                #optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

                # deep copy the model
                if phase == 'val' and epoch_loss < best_acc:
                    print('saving with loss of {}'.format(epoch_loss),
                          'improved over previous {}'.format(best_acc))
                    best_acc = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best loss: {:4f}'.format(float(best_acc)))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model


    model_ft = models.resnet50(pretrained=True)
    for param in model_ft.parameters():
        param.requires_grad = False #freeze
    #num_ftrs = model_ft.classifier[6].in_features
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 512)

    class multi_output_model(torch.nn.Module):
        def __init__(self, model_core,dd):
            super(multi_output_model, self).__init__()
        
            self.resnet_model = model_core
        
            self.x1 =  nn.Linear(512,256)
            nn.init.xavier_normal_(self.x1.weight)
        
            self.bn1 = nn.BatchNorm1d(256,eps = 1e-2)
            self.x2 =  nn.Linear(256,256)
            nn.init.xavier_normal_(self.x2.weight)
            self.bn2 = nn.BatchNorm1d(256,eps = 2e-1)
        
        
            #heads
            self.y1o = nn.Linear(256,cat_nodes)
            nn.init.xavier_normal_(self.y1o.weight)#
            self.y2o = nn.Linear(256,attr_nodes)
            nn.init.xavier_normal_(self.y2o.weight)
        
        
            self.d_out = nn.Dropout(dd)
        def forward(self, x):
       
            x = self.resnet_model(x)
        #x1 = F.relu(self.x1(x))
            x1 = self.bn1(F.relu(self.x1(x)))
            x1 = self.bn2(F.relu(self.x2(x1)))
        #x = F.relu(self.x2(x))
        #x1 = F.relu(self.x3(x))
        
        # heads
            y1o = F.softmax(self.y1o(x1),dim=1)
            y2o = torch.sigmoid(self.y2o(x1)) #should be sigmoid
            #y2o = F.softmax(self.y2o(x1),dim=1)
        
        #y1o = self.y1o(x1)
        #y2o = self.y2o(x1)
        
            return y1o, y2o


    dd = .1
    model_1 = multi_output_model(model_ft,dd)
    model_1 = model_1.to(device)
    #print(model_1)
    #print(model_1.parameters())    
    category_loss_func = torch.nn.CrossEntropyLoss()
    #attr_loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor([const.WEIGHT_ATTR_NEG, const.WEIGHT_ATTR_POS]).to(const.device))
    criterion = [category_loss_func,nn.BCELoss()]
    #criterion = [nn.CrossEntropyLoss(),nn.CrossEntropyLoss(),nn.CrossEntropyLoss(),nn.CrossEntropyLoss(),nn.BCELoss()]
    #criterion = [nn.CrossEntropyLoss(),nn.CrossEntropyLoss(),nn.CrossEntropyLoss(),nn.CrossEntropyLoss(),nn.MultiLabelSoftMarginLoss()]


    lrlast = .001
    lrmain = .0001


    optim = optim.Adam(
        [
            {"params":model_1.resnet_model.parameters(),"lr": lrmain},
            {"params":model_1.x1.parameters(), "lr": lrlast},
            {"params":model_1.x2.parameters(), "lr": lrlast},
            {"params":model_1.y1o.parameters(), "lr": lrlast},
            {"params":model_1.y2o.parameters(), "lr": lrlast},
        ],
        lr=lrmain)

    #optim = optim.Adam(model_1.parameters(),lr=lrmain)#, momentum=.9)
    # Observe that all parameters are being optimized
    optimizer_ft = optim

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
    model_ft1 = train_model(model_1, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=10)
    print("done training!")
    #saving model
    modelname = 'resnetFixed.pkl'
    torch.save(model_ft1.state_dict(), 'models/' + modelname)
