from libauc.models import resnet18
from libauc.losses import AUCMLoss

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO, Evaluator


from libauc.losses import AUCMLoss, CrossEntropyLoss
from libauc.optimizers import PESG, Adam
from libauc.models import densenet121 as DenseNet121
from libauc.datasets import CheXpert

import torch 
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore") 


data_flag = 'breastmnist'
download = True

NUM_EPOCHS = 3
BATCH_SIZE = 16
lr = 0.001

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

# Data Transformation
data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Resize((32, 32)), 
            
            transforms.Normalize(0.5, 0.5),
        ])

# load the data
train_dataset = DataClass(split='train', transform=data_transform, download=download)
test_dataset = DataClass(split='test', transform=data_transform, download=download)

pil_dataset = DataClass(split='train', download=download)

# encapsulate data into dataloader form
train_loader = torch_data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = torch_data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = torch_data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)




# BEST CROSS ENTROPY 
lr = 1e-5
weight_decay = 2e-5
model = resnet18(num_classes=2)
model = model.cuda()
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# training
best_val_auc = 0 
for epoch in range(201):
    train_losses = []
    for idx, data in enumerate(train_loader):
      train_data, train_labels = data
      train_data, train_labels  = train_data.cuda(), train_labels.cuda()
      y_pred = model(train_data)
      loss = criterion(y_pred, train_labels.squeeze().type(torch.LongTensor).cuda())
      train_losses.append(loss.item())
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    if epoch%5==0:
      print("Epoch : {:03d}  Train Loss : {:.5f}  ".format(epoch, np.mean(train_losses)), end='')
      model.eval()
      with torch.no_grad():    
          test_pred = []
          test_true = [] 
          test_losses = []
          for jdx, data in enumerate(test_loader):
              test_data, test_labels = data
              test_data = test_data.cuda()
              y_pred = model(test_data)
              test_pred.append(y_pred.cpu().detach().numpy())
              test_true.append(test_labels.numpy())
              test_losses.append(criterion(y_pred, test_labels.squeeze().type(torch.LongTensor).cuda()).item())

          test_true = np.concatenate(test_true)
          test_pred = np.concatenate(test_pred)
          val_auc_mean =  roc_auc_score(test_true.squeeze(), test_pred[:,1]) 
          print("Val Loss : {:.5f}   ".format(np.mean(test_losses)), end='')
          model.train()

          if best_val_auc < val_auc_mean:
              best_val_auc = val_auc_mean
              torch.save(model.state_dict(), 'ce_pretrained_model.pth')

          print ('BatchID= {}   Val_AUC={:.4f}   Best_Val_AUC={:.4f}'.format(
              idx, val_auc_mean, best_val_auc ))