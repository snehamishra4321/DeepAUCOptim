# Import all required modules
import os
import argparse
from tqdm import tqdm, trange
import numpy as np
import pdb
from PIL import Image
import time 
from collections import OrderedDict
import json
import random
import copy

from libauc.models import resnet18
from libauc.losses import AUCMLoss, CrossEntropyLoss, AUCM_MultiLabel_V1
from libauc.optimizers import PESG, Adam
from libauc.utils import ImbalancedDataGenerator
from libauc.sampler import DualSampler
from libauc.metrics import auc_roc_score


from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchsampler import ImbalancedDatasetSampler
from torch.utils.tensorboard import SummaryWriter

import medmnist
from medmnist import INFO, Evaluator
from medmnist import dataset as MedmnistDataset

import warnings
warnings.filterwarnings("ignore") 

print("INFO: IMPORTED LIBRARIES")

def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class MNIST_3D_Datasets(MedmnistDataset.MedMNIST3D):

    def __init__(self, dataset, transformation_list, as_rgb, target_transform):

        self.imgs = dataset.imgs
        self.labels = dataset.labels
        self.transform = transforms.Compose(transformation_list)
        self.as_rgb = as_rgb
        self.target_transform = target_transform

    def __getitem__(self, index):
        '''
        return: (without transform/target_transofrm)
            img: an array of 1x28x28x28 or 3x28x28x28 (if `as_RGB=True`), in [0,1]
            target: np.array of `L` (L=1 for single-label)
        '''
        img, target = self.imgs[index], self.labels[index].astype(int)

        img = np.stack([img/255.]*(3 if self.as_rgb else 1), axis=0)

        img = torch.tensor(img)

        # pdb.set_trace()
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target



# Set seed for reproducible results
SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

set_all_seeds(SEED)


def main(data_flag, output_root, num_epochs, gpu_ids, batch_size, download, model_flag, resize, as_rgb, model_path, run,  lr, margin, 
         optimizer_fn, epoch_decay, weight_decay, loss_fn, gamma, args, sampling_rate, momentum):

    milestones = [0.5 * num_epochs, 0.75 * num_epochs]

    info = INFO[data_flag]
    task = info['task']
    n_channels = 3 if as_rgb else info['n_channels']
    n_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])
    n_dim = DataClass(split='train', download=download, as_rgb=as_rgb).imgs.ndim

    str_ids = gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_ids[0])
    device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu') 
    
    output_root = os.path.join(output_root, data_flag)
    tensorboard_output_root = os.path.join(output_root, time.strftime("%y%m%d"))
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    if not os.path.exists(tensorboard_output_root):
        os.makedirs(tensorboard_output_root)

    # DATA TRANSFORMATIONS
    print('INFO: Preparing data...')

    transformation_list = []
    if n_dim <= 3:
        transformation_list.extend([transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Grayscale(3),
            transforms.ToTensor()])
        
        if resize:
            transformation_list.append(transforms.Resize((32, 32), interpolation=Image.NEAREST))

    transformation_list.extend([transforms.Resize((32, 32)), transforms.Normalize(mean=[.5], std=[.5])])
    data_transform = transforms.Compose(transformation_list)

    train_dataset = DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb)
    val_dataset = DataClass(split='val', transform=data_transform, download=download, as_rgb=as_rgb)
    test_dataset = DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb)

    if n_dim == 4:
        train_dataset.imgs = np.swapaxes(train_dataset.imgs, 1, 3) 
        val_dataset.imgs = np.swapaxes(val_dataset.imgs, 1, 3) 
        test_dataset.imgs = np.swapaxes(test_dataset.imgs, 1, 3) 
        train_dataset.imgs = torch.tensor(torch.from_numpy(train_dataset.imgs))
        train_dataset = MNIST_3D_Datasets(train_dataset, transformation_list=transformation_list, 
                                        as_rgb=train_dataset.as_rgb, target_transform=train_dataset.target_transform)
        val_dataset = MNIST_3D_Datasets(val_dataset, transformation_list=transformation_list, 
                                        as_rgb=train_dataset.as_rgb, target_transform=train_dataset.target_transform)
        test_dataset = MNIST_3D_Datasets(test_dataset, transformation_list=transformation_list, 
                                        as_rgb=train_dataset.as_rgb, target_transform=train_dataset.target_transform)
    
    # sampler = DualSampler(train_dataset, batch_size, sampling_rate=sampling_rate)
    # sampler = ImbalancedDatasetSampler(train_dataset)

    # pdb.set_trace()
    
    train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                # sampler=sampler,
                                shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    val_loader = data.DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    
    

    print('INFO: Building and training model...')
    
    
    if model_flag == 'resnet18':
        model =  resnet18(pretrained=False, num_classes=n_classes) if resize else ResNet18(in_channels=n_channels, num_classes=n_classes)
    elif model_flag == 'resnet50':
        model =  resnet50(pretrained=False, num_classes=n_classes) if resize else ResNet50(in_channels=n_channels, num_classes=n_classes)
    # else:
    #     raise NotImplementedError
    
    # model = Resnet18(pretrained=False, num_classes=n_classes, in_channels=n_channels)

    if n_dim == 4:
        model.conv1 = torch.nn.Conv2d(train_dataset.imgs.shape[1], 64, kernel_size=7, stride=2, padding=3, bias=False)

    model = model.to(device)

    global test_evaluator

    train_evaluator = medmnist.Evaluator(data_flag, 'train')
    val_evaluator = medmnist.Evaluator(data_flag, 'val')
    test_evaluator = medmnist.Evaluator(data_flag, 'test')

    if loss_fn == 'CE':
        if task == "multi-label, binary-class":
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
    elif loss_fn == 'AUCM':
        criterion = AUCMLoss()
    elif loss_fn == 'AUCM_Multilabel':
        criterion = AUCM_MultiLabel_V1()
    print("Loss Function Used : {}".format(criterion))

    # IN CASE OF PRETRAINED
    # if model_path is not None:
    #     model.load_state_dict(torch.load(model_path, map_location=device)['net'], strict=True)
    #     train_metrics = test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run, output_root)
    #     val_metrics = test(model, val_evaluator, val_loader, task, criterion, device, run, output_root)
    #     test_metrics = test(model, test_evaluator, test_loader, task, criterion, device, run, output_root)

    #     print('train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2]) + \
    #           'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2]) + \
    #           'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2]))

    if num_epochs == 0:
        return

    if optimizer_fn == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_fn == 'PESG':
        optimizer = PESG(model, 
                 loss_fn=criterion, 
                 lr=lr, 
                 momentum=momentum, 
                 margin=margin, 
                 epoch_decay=epoch_decay, 
                 weight_decay=weight_decay)
    print("Optimizer Used : {}".format(optimizer.__str__()[:5]))

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma, verbose=False)

    logs = ['loss', 'auc', 'acc']
    train_logs = ['train_'+log for log in logs]
    val_logs = ['val_'+log for log in logs]
    test_logs = ['test_'+log for log in logs]
    log_dict = OrderedDict.fromkeys(train_logs+val_logs+test_logs, 0)
    
    out_filename = 'BatchSz_' + str(batch_size) +'_LR_' + str(lr) + '_Optimizer_'+str(optimizer_fn)+'_LossFn_'+str(loss_fn)+'_Margin_'+str(margin)+'_EpochDecay_'+str(epoch_decay)+'_WeightDecay_'+str(weight_decay)
    writer = SummaryWriter(log_dir=os.path.join(tensorboard_output_root, 'Tensorboard_Results', out_filename))

    best_auc = 0
    best_epoch = 0
    best_model = model

    global iteration, epoch_log
    iteration = 0
    epoch_log = {}

    for epoch in trange(num_epochs):        

        train_loss = train(model, train_loader, test_loader, task, criterion, optimizer, device, writer, loss_fn)
        
        train_metrics = test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run, None, loss_fn)
        val_metrics = test(model, val_evaluator, val_loader, task, criterion, device, run, None,  loss_fn)
        test_metrics = test(model, test_evaluator, test_loader, task, criterion, device, run, None,  loss_fn)
        
        scheduler.step()
        
        for i, key in enumerate(train_logs):
            log_dict[key] = train_metrics[i]
        for i, key in enumerate(val_logs):
            log_dict[key] = val_metrics[i]
        for i, key in enumerate(test_logs):
            log_dict[key] = test_metrics[i]

        for key, value in log_dict.items():
            writer.add_scalar(key.split('_')[0] + '/' + key.split('_')[1], value, iteration)
            
        epoch_log[epoch] = dict(log_dict)

        # CHOOSE WHICH METRIC TO IDENTIFY BEST MODEL - ALWAYS GO WITH VAL
        if args.based_on == 'val':
            cur_auc = val_metrics[1]
        else:
            cur_auc = test_metrics[1]

        if cur_auc > best_auc:
            best_epoch = epoch
            best_auc = cur_auc
            best_model = copy.deepcopy(model)
            print('\ncur_best_auc:', best_auc)
            print('cur_best_epoch', best_epoch)

    # EVALUATE THE BEST MODEL

    train_metrics = test(best_model, train_evaluator, train_loader_at_eval, task, criterion, device, run,  None, loss_fn)
    val_metrics = test(best_model, val_evaluator, val_loader, task, criterion, device, run, None, loss_fn)
    test_metrics = test(best_model, test_evaluator, test_loader, task, criterion, device, run,  None,loss_fn)

    for i, key in enumerate(train_logs):
        log_dict[key] = train_metrics[i]
    for i, key in enumerate(val_logs):
        log_dict[key] = val_metrics[i]
    for i, key in enumerate(test_logs):
        log_dict[key] = test_metrics[i]

    train_log = 'train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2])
    val_log = 'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2])
    test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2])

    log = '%s\n' % (data_flag) + train_log + val_log + test_log
    print(log)
    log_results(args, best_metrics=log_dict, best_metric_filename=os.path.join(output_root,'Best_metrics_of_each_run.txt'), 
                epoch_log=epoch_log, epoch_log_fname=os.path.join(output_root, 'Epoch_Logs',out_filename))
                
    writer.close()


def train(model, train_loader, test_loader, task, criterion, optimizer, device, writer, loss_fn):
    total_loss = []
    global iteration

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):

        if len(train_loader.dataset.imgs.shape)==4:
            inputs = inputs.squeeze()

        optimizer.zero_grad()
        if loss_fn == 'CE':
            outputs = model(inputs.to(device))
        elif loss_fn == 'AUCM':
            outputs = torch.sigmoid(model(inputs.to(device)))  # OG
        elif loss_fn == 'AUCM_Multilabel':
            # pdb.set_trace()
            softmax = torch.nn.Softmax()
            outputs = softmax(model(inputs.to(device)))
            
                
            # # outputs = nn.Softmax(model(inputs.to(device)))
            # m = nn.Softmax()
            # outputs = m(model(inputs.to(device)))

        if task == 'multi-label, binary-class':
            # pdb.set_trace()
            targets = targets.to(torch.float32).to(device)
            loss = criterion(outputs, targets)
        else:
            targets = torch.squeeze(targets, 1).long().to(device)
            loss = criterion(outputs, targets)

        total_loss.append(loss.item())
        # print(loss.item())
        writer.add_scalar('train_loss_logs', loss.item(), iteration)
        iteration += 1
        loss.backward()
        optimizer.step()

        if iteration%30 == 0 :
            test_metrics = test(model, test_evaluator, test_loader, task, criterion, device, run,  None,loss_fn)
            test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2])
            print(test_log)
    
    epoch_loss = sum(total_loss)/len(total_loss)
    return epoch_loss

def test(model, evaluator, data_loader, task, criterion, device, run, save_folder=None, loss_fn=None):

    model.eval()
    
    total_loss = []
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):

            if len(data_loader.dataset.imgs.shape)==4:
                inputs = inputs.squeeze()
            try:
                outputs = model(inputs.to(device))
            except:
                outputs = model(inputs.float().to(device))
            
            if loss_fn == 'CE':
                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32).to(device)
                    loss = criterion(outputs, targets)
                    m = nn.Sigmoid()
                    outputs = m(outputs).to(device)
                else:
                    targets = torch.squeeze(targets, 1).long().to(device)
                    loss = criterion(outputs, targets)
                    m = nn.Softmax(dim=1)
                    outputs = m(outputs).to(device)
                    targets = targets.float().resize_(len(targets), 1)
            elif loss_fn == 'AUCM':
                # pdb.set_trace()
                targets = torch.squeeze(targets, 1).long().to(device)
                loss = criterion(outputs, targets)
                outputs = outputs.to(device)
                targets = targets.float().resize_(len(targets), 1)
            elif loss_fn == 'AUCM_Multilabel':
                # pdb.set_trace()
                targets = torch.squeeze(targets, 1).long().to(device)
                loss = criterion(outputs, targets)  
                outputs = outputs.to(device)
                # targets = targets.float().resize_(len(targets), 1)

            total_loss.append(loss.item())
            y_score = torch.cat((y_score, outputs), 0)

        y_score = y_score.detach().cpu().numpy()
        auc, acc = evaluator.evaluate(y_score, save_folder=None, run=run)

        test_loss = sum(total_loss) / len(total_loss)

        return [test_loss, auc, acc]


def log_results(args, best_metrics, best_metric_filename, epoch_log, epoch_log_fname:str):
    # Write Best Metrics
    data = vars(args)
    data.update(best_metrics)
    try:
        with open(best_metric_filename, 'r', encoding="utf8") as file:
            file_data = json.loads(file.read())
            file_data.append(data)
    except:
        file_data = [data]

    with open(best_metric_filename,'w', encoding="utf8") as file:
        json.dump(file_data, file)

    # Write Epoch Logs
    if not os.path.exists(os.path.dirname(epoch_log_fname)):
        os.makedirs(os.path.dirname(epoch_log_fname))
    with open(epoch_log_fname,'w') as file:
        file.write(str(epoch_log))


if __name__== '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST2D')
    
    parser.add_argument('--data_flag',
                        default='breastmnist',
                        type=str)
    parser.add_argument('--output_root',
                        default='./output',
                        help='output root, where to save models and results',
                        type=str)
    parser.add_argument('--num_epochs',
                        default=100,
                        help='num of epochs of training, the script would only test model if set num_epochs to 0',
                        type=int)
    parser.add_argument('--gpu_ids',
                        default='0',
                        type=str)
    parser.add_argument('--batch_size',
                        default=128,
                        type=int)
    parser.add_argument('--download',
                        action="store_true")
    parser.add_argument('--resize',
                        # help='resize images of size 28x28 to 224x224',
                        help='resize images of size 28x28 to 32x32. By default the image should be resized',
                        action="store_false")
    parser.add_argument('--as_rgb',
                        help='convert the grayscale image to RGB',
                        action="store_true")
    parser.add_argument('--model_path',
                        default=None,
                        help='root of the pretrained model to test',
                        type=str)
    parser.add_argument('--model_flag',
                        default='resnet18',
                        choices=['resnet18', 'resnet50'],
                        help='choose backbone from resnet18, resnet50',
                        type=str)
    parser.add_argument('--loss',
                        default='AUCM_Multilabel',
                        help='Loss function to train the model',
                        choices=['AUCM', 'CE', 'AUCM_Multilabel'],
                        type=str)
    parser.add_argument('--lr',
                        default=0.001,
                        help='Learning rate',
                        type=float)
    parser.add_argument('--margin',
                        default=1.0,
                        help='Margin for PESG loss',
                        type=float)
    parser.add_argument('--optimizer',
                        default='PESG',
                        help='Optimizer',
                        choices=['Adam','PESG'],
                        type=str)
    parser.add_argument('--epoch_decay',
                        default=2e-3,
                        help='Epoch decay for PESG Loss',
                        type=float)
    parser.add_argument('--momentum',
                        default=0.9,
                        help='momentum parameter for PESG optimizer',
                        type=float)
    parser.add_argument('--weight_decay',
                        default=1e-5,
                        help='Weight decay parameter for optimization',
                        type=float)
    parser.add_argument('--gamma',
                        default=0.1,
                        help='Gamma parameter for Multistep LR Scheduler',
                        type=float)
    parser.add_argument('--sampling_rate',
                        default=0.0,
                        help='The oversampling ratio for the positive minority class',
                        type=float)
    parser.add_argument('--run',
                        default='model1',
                        help='to name a standard evaluation csv file, named as {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv',
                        type=str)
    parser.add_argument('--based_on',
                        default='val',
                        help='How to select best models - Test or Val?',
                        type=str)
    parser.add_argument('--modify_datasets',
                        default='',
                        choices=['halve_val', 'double_val'],
                        help='To change the size of validation and training dataset',
                        type=str)
    

    args = parser.parse_args()
    data_flag = args.data_flag
    output_root = args.output_root
    num_epochs = args.num_epochs
    gpu_ids = args.gpu_ids
    batch_size = args.batch_size
    download = args.download
    model_flag = args.model_flag
    resize = args.resize
    as_rgb = args.as_rgb
    model_path = args.model_path
    run = args.run
    lr = args.lr
    margin = args.margin
    optimizer = args.optimizer
    epoch_decay = args.epoch_decay
    weight_decay = args.weight_decay
    loss = args.loss
    gamma = args.gamma

    # pdb.set_trace()
    
    main(data_flag, output_root, num_epochs, gpu_ids, batch_size, download, model_flag, resize, as_rgb, model_path, 
         run, lr, margin, optimizer, epoch_decay, weight_decay, loss, gamma, args, args.sampling_rate, args.momentum)