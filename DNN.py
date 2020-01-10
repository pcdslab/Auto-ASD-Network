import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from functools import reduce
from sklearn.impute import SimpleImputer
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import pyprind
import sys
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold, StratifiedKFold
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from scipy import stats
from sklearn import tree
import numpy.ma as ma # for masked arrays
import csv
from sklearn.feature_selection import SelectFromModel
import random
import warnings
import shutil
warnings.filterwarnings('ignore')


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if not os.path.exists('results'):
    os.makedirs('results')

site = sys.argv[1]

data_main_path = './dl'#path to time series data
flist = os.listdir(data_main_path)
df_labels = pd.read_csv('./pheno/Phenotypic_V1_0b_preprocessed1.csv')#path 

df_labels.DX_GROUP = df_labels.DX_GROUP.map({1: 1, 2:0})

labels = {}
for row in df_labels.iterrows():
    file_id = row[1]['FILE_ID']
    y_label = row[1]['DX_GROUP']
    if file_id == 'no_filename':
        continue
    assert(file_id not in labels)
    labels[file_id] = y_label




def get_label(filename):
    f_split = filename.split('_')
    if f_split[3] == 'rois':
        key = '_'.join(f_split[0:3]) 
    else:
        key = '_'.join(f_split[0:2])
    assert (key in labels)
    
    return labels[key]
    

def get_corr_data(filename,pathtodata):
    df = pd.read_csv(os.path.join(pathtodata, filename), sep='\t')
    with np.errstate(invalid="ignore"):
        corr = np.nan_to_num(np.corrcoef(df.T))
        mask = np.invert(np.tri(corr.shape[0], k=-1, dtype=bool))
        m = ma.masked_where(mask == 1, mask)
        features = ma.masked_where(m, corr).compressed()
        return features


def confusion(g_turth,predictions):
    tn, fp, fn, tp = confusion_matrix(g_turth,predictions).ravel()
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    sensitivity = (tp)/(tp+fn)
    specificty = (tn)/(tn+fp)
    return accuracy,sensitivity,specificty


flist2 = flist.copy()


pbar=pyprind.ProgBar(len(flist))
all_corrcc200 = {}
for f in flist2:
    f_split = f.split('_')
    if f_split[3] == 'rois':
        key = '_'.join(f_split[0:3]) #,f_split[2]])
    else:
        key = '_'.join(f_split[0:2])


    lab = get_label(f)
    all_corrcc200[f] = (get_corr_data(f,data_main_path),lab)#,get_corr_data_firstthreequarters(f,data_main_path),get_corr_data_lastthreequarters(f,data_main_path))#(get_DFC(f,2), lab)
    pbar.update()
print('Correlation computations finished')



class CC200Dataset(Dataset):
    def __init__(self, datay, samples_list,mode,augmentation):
     
        self.data = datay.copy()
        self.mode = mode    


        self.flist = [f for f in samples_list]
        self.labels = np.array([self.data[f][1] for f in self.flist])
        
        current_flist = np.array(self.flist.copy())
        current_lab0_flist = current_flist[self.labels == 0]
        current_lab1_flist = current_flist[self.labels == 1]

        self.num_data = len(self.flist)
        if augmentation:
            self.num_data = 2* len(self.flist)
            self.neighbors = {}
            for f in self.flist:
                label = self.data[f][1]
                candidates = (set(current_lab0_flist) if label == 0 else set(current_lab1_flist))
                candidates.remove(f)
                self.neighbors[f] = list(candidates).copy()#[item[1] for item in sim_list[:num_neighbs]]#list(candidates)#[item[1] for item in sim_list[:num_neighbs]]

    def __getitem__(self, index):
        if index < len(self.flist):
            fname = self.flist[index]
            data = self.data[fname][0].copy() 
            label = [self.labels[index]]#(self.labels[index],)        
            return torch.FloatTensor(data), torch.FloatTensor(label)
        
        else:
            f1 = self.flist[index % len(self.flist)]
            d1, y1 = self.data[f1][0], self.data[f1][1]
            
            
            f2 = np.random.choice(self.neighbors[f1])
            d2, y2 = self.data[f2][0], self.data[f2][1]
            
            f3 = np.random.choice(self.neighbors[f1])
            d3,y3 =  self.data[f3][0], self.data[f3][1]
            
            assert y1 == y3
            assert y2 == y3
            assert y1 == y2
            r = np.random.uniform(low=0, high=1)
            label = (y1,)
            data = r*d3 +  (1-r)*d2
            return torch.FloatTensor(data), torch.FloatTensor(label)
            
    def __len__(self):
        return self.num_data

	


def get_loader(data, samples_list,batch_size, mode ,augmentation):
    """Build and return data loader."""
    torch.manual_seed(200)
    if mode == 'train':
        shuffle = True
    else:
        shuffle = False
        augmentation=False
    dataset = CC200Dataset(data, samples_list,mode,augmentation)

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
  
    return data_loader

def extension(samps):
    new_samps = []
    for i in samps:
        new_samps.append(i+'_rois_cc200.1D')
    new_samps = np.array(new_samps)
    return new_samps

def extensioncc400(samps):
    new_samps = []
    for i in samps:
        new_samps.append(i+'_rois_cc400.1D')
    new_samps = np.array(new_samps)
    return new_samps




class FeedForward(nn.Module):
    def __init__(self, num_inputs=990, 
                 num_hidden=200,
                 num_classes=2):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(num_inputs,int(num_hidden/2))
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(int(num_hidden/2),int(num_hidden/4))
        self.fc3 = nn.Linear(int(num_hidden/4),num_classes)
         
    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.relu(out1)
        out = self.fc2(out2)
        out3 = self.relu(out)
        out3 = self.fc3(out3)###(out)
        return out, out3



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, epoch, train_loader,optimizer,criterion):
    model.train()
    train_losses = []
    for i,(batch_x,batch_y) in enumerate(train_loader):

        batch_y=batch_y.long()
        batch_y=batch_y.flatten()
        dataytrain, target = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        
        out1,out = model(dataytrain)
        loss = criterion(out, target)
        train_losses.append([loss.item()]) 
        
        loss.backward()
        optimizer.step()

    return train_losses    

def test(model,test_loader):
    test_loss, n_test, correct = 0.0, 0, 0
    total,sensi,speci,tot0,tot1 = 0, 0, 0,0,0

    all_predss=[]
    y_true, y_pred = [], []
    with torch.no_grad():
        model.eval()
        for i,(batch_x,batch_y) in enumerate(test_loader, 1):
            y_arr = np.array(batch_y, dtype=np.int32)
            
            datayytest = batch_x.to(device)
            hidden, logits = model(datayytest)
            _, predicted = torch.max(logits.data,1)
            
            

            total += batch_y.size(0)                    # Increment the total count
            
            correct += (predicted.cpu().numpy() == y_arr.flatten()).sum()     # Increment the correct count
            for inum in range(len(predicted.cpu().numpy())):
                if y_arr.flatten()[inum]==1:
                    tot0+=1
                    if predicted.cpu().numpy()[inum]==1:
                        sensi+=1
                else:
                    tot1+=1
                    if predicted.cpu().numpy()[inum]==0:
                        speci+=1

    return np.array([100 * correct / total, 100*sensi/tot0, 100*speci/tot1])


def get_hidden_taki(model_, samples):
    
    y_arr = []
    with torch.no_grad():
        model_.eval()
        bottles = []#np.empty((0,n_lat))#np.array([])
        for _i in samples:
            data_to_dv = torch.FloatTensor(all_corr[_i][0])#[which_model-1])
            y_arr.extend([all_corr[_i][1]])
            datass = data_to_dv.to(device)
            out, bottleneck = model_(datass)
            bottles.append(np.asarray(out.cpu()))
            
            #out.detach().cpu().numpy()
    y_arr = np.array(y_arr)
    
    return np.array(bottles),y_arr



def train_test(site,augment):
    resres=[]
    resresother = []
    for k in range(5):
        train_samples = np.genfromtxt('./subjects/'+site+'/train_sub'+str(k),dtype='str')
        test_samples = np.genfromtxt('./subjects/'+site+'/test_sub'+str(k),dtype='str')
        

        train_samples=extension(train_samples)#extensioncc400(train_samples)
        test_samples=extension(test_samples)#extensioncc400(test_samples)
        
        train_loader=get_loader(all_corr, train_samples,batch_size,'train',augment)
        test_loader=get_loader(all_corr, test_samples,batch_size, 'test',augment)
        model = FeedForward(num_inputs=19900, num_hidden=int(19900/2),num_classes=2)
        model.to(device)
        criterion_ = nn.CrossEntropyLoss()
        optimizer_ = torch.optim.SGD(model.parameters(), lr=0.001)

        for epoch in range(1, num_epochs+1):    
            train_losses = train(model, epoch, train_loader,optimizer_,criterion_)

        new_features,new_label = get_hidden_taki(model, train_samples)
        new_features_test,new_label_test = get_hidden_taki(model, test_samples)
        
        new_label=new_label.reshape((len(new_label),1))
        new_label_test=new_label_test.reshape((len(new_label_test),1))
        new_features = np.concatenate((new_features, new_label), axis=1)
        new_features_test = np.concatenate((new_features_test, new_label_test), axis=1)
        df = pd.DataFrame(new_features)
        df = df.rename(columns={new_features.shape[1]-1: 'class'})
        df.to_csv('./features'+site+'/'+str(k)+'_'+str(augment)+'_train.csv',index_label=False)

        df2 = pd.DataFrame(new_features_test)
        df2 = df2.rename(columns={new_features_test.shape[1]-1: 'class'})
        df2.to_csv('./features'+site+'/'+str(k)+'_'+str(augment)+'_test.csv',index_label=False)
        
        
        #######################################

        resres.append(test(model, test_loader))
        print("fold:",k,":",test(model, test_loader))
         
    
#    print("Average result of all folds:",np.around(np.mean(resres,axis=0),decimals=2))

    Description = 'MLP'
    if augment == True:
        Description +='-DA'
   
    print("\nAverage result "+Description+ "\nAccuracy Sensitivity Specificity:")
    print(np.around(np.mean(resres,axis=0),decimals=2),"\n--------------------------------\n")

    with open("./results/"+Description, 'w') as file: 
        file.write("Accuracy Sensitivity Specificity\n")
        np.savetxt(file,[np.mean(resres,axis=0)],fmt='%1.2f')
        
p_fold=5
flist = flist2.copy()#np.array(sorted(os.listdir(data_main_path)))
flist = np.array(flist)
batch_size = 4
num_epochs =60
np.random.seed(19)
random.seed(19)
all_corr = all_corrcc200

if os.path.isdir("features"+site):
    shutil.rmtree("features"+site, ignore_errors=False, onerror=None)
os.makedirs("features"+site)

train_test(site,True)

train_test(site,False)
