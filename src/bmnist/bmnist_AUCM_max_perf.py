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


# data_flag = 'pathmnist'
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

# # preprocessing
# data_transform = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Grayscale(3),
#     transforms.Resize((28, 28)), 
#     transforms.Normalize(mean=[.5], std=[.5])
# ])

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


lr = 0.000001
lr = 0.005 # using smaller learning rate is better
epoch_decay = 2e-4
weight_decay = 1e-5
margin = 1.0

model = resnet18(num_classes=2)
model = model.cuda()
# criterion = nn.CrossEntropyLoss()  
# optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = AUCMLoss()
optimizer = PESG(model, 
                 loss_fn=criterion, 
                 lr=lr, 
                 margin=margin, 
                 epoch_decay=epoch_decay, 
                 weight_decay=weight_decay)
CE_loss = nn.CrossEntropyLoss()
# training
best_val_auc = 0 
for epoch in range(201):
    train_losses = []
    for idx, data in enumerate(train_loader):
      train_data, train_labels = data
      train_data, train_labels  = train_data.cuda(), train_labels.cuda()
      y_pred = model(train_data)
      y_pred = torch.sigmoid(y_pred)
      loss = criterion(y_pred, train_labels.squeeze().type(torch.LongTensor).cuda())
      train_losses.append(loss.item()/len(data))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
        
      # validation  
    #   if idx % 10 == 0:
    
    if epoch%5==0:
      print("Epoch : {:03d}  Train Loss : {:.5f} ".format(epoch, np.mean(train_losses)), end='')
      model.eval()
      with torch.no_grad():    
          test_pred = []
          test_true = [] 
          test_losses = []
          test_CE_losses = []
          for jdx, data in enumerate(test_loader):
              test_data, test_labels = data
              test_data = test_data.cuda()
              y_pred = model(test_data)
              test_pred.append(y_pred.cpu().detach().numpy())
              test_true.append(test_labels.numpy())
              test_losses.append(criterion(y_pred, test_labels.squeeze().type(torch.LongTensor).cuda()).item() / len(data))
              test_CE_losses.append(CE_loss(y_pred, test_labels.squeeze().type(torch.LongTensor).cuda()).cpu())

          test_true = np.concatenate(test_true)
          test_pred = np.concatenate(test_pred)
          val_auc_mean =  roc_auc_score(test_true.squeeze(), test_pred[:,1]) 
          print("Val Loss : {:.5f}   Val CE Loss : {:.5f}  ".format(np.mean(test_losses), np.mean(test_CE_losses)), end = '')
          model.train()

          if best_val_auc < val_auc_mean:
              best_val_auc = val_auc_mean
              torch.save(model.state_dict(), 'ce_pretrained_model.pth')

          print ('BatchID= {}   Val_AUC={:.4f}   Best_Val_AUC={:.4f}'.format(
              idx, val_auc_mean, best_val_auc ))
          

        #   Using downloaded and verified file: C:\Users\Hari\.medmnist\breastmnist.npz
# Using downloaded and verified file: C:\Users\Hari\.medmnist\breastmnist.npz
# Using downloaded and verified file: C:\Users\Hari\.medmnist\breastmnist.npz
# Epoch : 000  Train Loss : 0.04016 Val Loss : 1.87842   Val CE Loss : 0.87590  BatchID= 34   Val_AUC=0.6456   Best_Val_AUC=0.6456
# Epoch : 005  Train Loss : 0.08333 Val Loss : 38.18521   Val CE Loss : 1.13688  BatchID= 34   Val_AUC=0.6527   Best_Val_AUC=0.6527
# Epoch : 010  Train Loss : 0.07066 Val Loss : 0.96823   Val CE Loss : 0.68566  BatchID= 34   Val_AUC=0.7379   Best_Val_AUC=0.7379
# Epoch : 015  Train Loss : 0.07180 Val Loss : 1.11744   Val CE Loss : 0.73136  BatchID= 34   Val_AUC=0.7316   Best_Val_AUC=0.7379
# Epoch : 020  Train Loss : 0.05939 Val Loss : 0.71424   Val CE Loss : 0.70164  BatchID= 34   Val_AUC=0.7629   Best_Val_AUC=0.7629
# Epoch : 025  Train Loss : 0.05603 Val Loss : 0.70176   Val CE Loss : 0.68684  BatchID= 34   Val_AUC=0.7755   Best_Val_AUC=0.7755
# Epoch : 030  Train Loss : 0.03941 Val Loss : 1.34463   Val CE Loss : 0.71883  BatchID= 34   Val_AUC=0.7926   Best_Val_AUC=0.7926
# Epoch : 035  Train Loss : 0.06127 Val Loss : 1.27371   Val CE Loss : 0.72547  BatchID= 34   Val_AUC=0.7141   Best_Val_AUC=0.7926
# Epoch : 040  Train Loss : 0.04716 Val Loss : 0.73208   Val CE Loss : 0.72746  BatchID= 34   Val_AUC=0.7845   Best_Val_AUC=0.7926
# Epoch : 045  Train Loss : 0.04905 Val Loss : 0.93399   Val CE Loss : 0.69785  BatchID= 34   Val_AUC=0.7847   Best_Val_AUC=0.7926
# Epoch : 050  Train Loss : 0.03547 Val Loss : 1.31716   Val CE Loss : 0.71247  BatchID= 34   Val_AUC=0.8108   Best_Val_AUC=0.8108
# Epoch : 055  Train Loss : 0.04448 Val Loss : 2.75855   Val CE Loss : 0.69856  BatchID= 34   Val_AUC=0.6748   Best_Val_AUC=0.8108
# Epoch : 060  Train Loss : 0.05102 Val Loss : 3.86967   Val CE Loss : 0.85014  BatchID= 34   Val_AUC=0.6959   Best_Val_AUC=0.8108
# Epoch : 065  Train Loss : 0.04000 Val Loss : 0.73258   Val CE Loss : 0.72935  BatchID= 34   Val_AUC=0.7957   Best_Val_AUC=0.8108
# Epoch : 070  Train Loss : 0.03564 Val Loss : 0.91404   Val CE Loss : 0.72355  BatchID= 34   Val_AUC=0.7682   Best_Val_AUC=0.8108
# Epoch : 075  Train Loss : 0.03334 Val Loss : 67.33063   Val CE Loss : 0.77133  BatchID= 34   Val_AUC=0.6293   Best_Val_AUC=0.8108
# Epoch : 080  Train Loss : 0.05769 Val Loss : 1.37564   Val CE Loss : 0.74568  BatchID= 34   Val_AUC=0.7634   Best_Val_AUC=0.8108
# Epoch : 085  Train Loss : 0.04421 Val Loss : 1.63566   Val CE Loss : 0.69185  BatchID= 34   Val_AUC=0.7571   Best_Val_AUC=0.8108
# Epoch : 090  Train Loss : 0.04412 Val Loss : 1.58405   Val CE Loss : 0.66897  BatchID= 34   Val_AUC=0.7726   Best_Val_AUC=0.8108
# Epoch : 095  Train Loss : 0.05243 Val Loss : 1.00083   Val CE Loss : 0.71281  BatchID= 34   Val_AUC=0.7876   Best_Val_AUC=0.8108
# Epoch : 100  Train Loss : 0.05320 Val Loss : 2.12024   Val CE Loss : 0.88817  BatchID= 34   Val_AUC=0.6794   Best_Val_AUC=0.8108
# Epoch : 105  Train Loss : 0.04845 Val Loss : 1.45531   Val CE Loss : 0.70558  BatchID= 34   Val_AUC=0.7195   Best_Val_AUC=0.8108
# Epoch : 110  Train Loss : 0.04324 Val Loss : 3.40647   Val CE Loss : 0.69672  BatchID= 34   Val_AUC=0.7439   Best_Val_AUC=0.8108
# Epoch : 115  Train Loss : 0.03450 Val Loss : 1.28976   Val CE Loss : 0.67647  BatchID= 34   Val_AUC=0.8206   Best_Val_AUC=0.8206
# Epoch : 120  Train Loss : 0.03293 Val Loss : 3.35686   Val CE Loss : 0.83178  BatchID= 34   Val_AUC=0.7471   Best_Val_AUC=0.8206
# Epoch : 125  Train Loss : 0.02693 Val Loss : 1.92084   Val CE Loss : 0.72502  BatchID= 34   Val_AUC=0.8104   Best_Val_AUC=0.8206
# Epoch : 130  Train Loss : 0.03237 Val Loss : 2.07734   Val CE Loss : 0.73839  BatchID= 34   Val_AUC=0.8394   Best_Val_AUC=0.8394
# Epoch : 135  Train Loss : 0.02997 Val Loss : 2.50633   Val CE Loss : 0.72849  BatchID= 34   Val_AUC=0.7726   Best_Val_AUC=0.8394
# Epoch : 140  Train Loss : 0.02594 Val Loss : 3.20529   Val CE Loss : 0.74481  BatchID= 34   Val_AUC=0.8810   Best_Val_AUC=0.8810
# Epoch : 145  Train Loss : 0.02375 Val Loss : 2.06508   Val CE Loss : 0.70285  BatchID= 34   Val_AUC=0.7932   Best_Val_AUC=0.8810
# Epoch : 150  Train Loss : 0.02453 Val Loss : 10.11439   Val CE Loss : 0.89552  BatchID= 34   Val_AUC=0.6535   Best_Val_AUC=0.8810
# Epoch : 155  Train Loss : 0.02739 Val Loss : 4.97180   Val CE Loss : 0.90483  BatchID= 34   Val_AUC=0.7421   Best_Val_AUC=0.8810
# Epoch : 160  Train Loss : 0.02276 Val Loss : 1.77604   Val CE Loss : 0.69197  BatchID= 34   Val_AUC=0.8548   Best_Val_AUC=0.8810
# Epoch : 165  Train Loss : 0.02121 Val Loss : 1.30612   Val CE Loss : 0.73622  BatchID= 34   Val_AUC=0.8816   Best_Val_AUC=0.8816
# Epoch : 170  Train Loss : 0.01244 Val Loss : 1.95634   Val CE Loss : 0.77152  BatchID= 34   Val_AUC=0.8651   Best_Val_AUC=0.8816
# Epoch : 175  Train Loss : 0.02923 Val Loss : 7.37326   Val CE Loss : 0.71380  BatchID= 34   Val_AUC=0.6525   Best_Val_AUC=0.8816
# Epoch : 180  Train Loss : 0.01326 Val Loss : 4.98931   Val CE Loss : 1.12494  BatchID= 34   Val_AUC=0.7116   Best_Val_AUC=0.8816
# Epoch : 185  Train Loss : 0.01312 Val Loss : 2.17656   Val CE Loss : 0.70883  BatchID= 34   Val_AUC=0.8626   Best_Val_AUC=0.8816
# Epoch : 190  Train Loss : 0.01886 Val Loss : 2.28042   Val CE Loss : 1.14534  BatchID= 34   Val_AUC=0.7421   Best_Val_AUC=0.8816
# Epoch : 195  Train Loss : 0.01207 Val Loss : 2.38285   Val CE Loss : 0.67144  BatchID= 34   Val_AUC=0.8317   Best_Val_AUC=0.8816
# Epoch : 200  Train Loss : 0.01185 Val Loss : 4.05147   Val CE Loss : 0.73717  BatchID= 34   Val_AUC=0.7899   Best_Val_AUC=0.8816