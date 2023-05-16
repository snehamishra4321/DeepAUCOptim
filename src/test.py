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

import torchio as tio

from libauc.models import resnet18
from libauc.losses import AUCMLoss, CrossEntropyLoss, pAUCLoss  
from libauc.optimizers import PESG, Adam, SOPA
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

from sklearn.metrics import accuracy_score as acc_score

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
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def halve_val_dset(train_dataset, val_dataset):
    val_pos_indices = np.where(val_dataset.labels==1)[0]
    val_neg_indices = np.where(val_dataset.labels==0)[0]
    val_pos_indices_to_transfer = np.random.choice(val_pos_indices, size = len(val_pos_indices)//2)
    val_neg_indices_to_transfer = np.random.choice(val_neg_indices, size = len(val_neg_indices)//2)
    val_indices_to_transfer = np.concatenate((val_pos_indices_to_transfer, val_neg_indices_to_transfer))
    val_indices_not_transfer = [i for i in range(val_dataset.imgs.shape[0]) if i not in val_indices_to_transfer]

    train_dataset_v1 = copy.deepcopy(train_dataset)
    train_dataset_v1.imgs = np.concatenate((train_dataset_v1.imgs, val_dataset.imgs[val_indices_to_transfer,:,:]), axis=0)
    train_dataset_v1.labels = np.concatenate((train_dataset.labels, val_dataset.labels[val_indices_to_transfer]), axis=0)

    val_dataset_v1 = copy.deepcopy(val_dataset)
    val_dataset_v1.imgs = val_dataset.imgs[val_indices_not_transfer,:,:]
    val_dataset_v1.labels = val_dataset.labels[val_indices_not_transfer]

    return train_dataset_v1, val_dataset_v1


def double_val_dset(train_dataset, val_dataset):
    """
    Makes Validation set 1.5 times
    """
    train_pos_indices = np.where(train_dataset.labels==1)[0]
    train_neg_indices = np.where(train_dataset.labels==0)[0]

    val_pos_indices = np.where(val_dataset.labels==1)[0]
    val_neg_indices = np.where(val_dataset.labels==0)[0]

    train_pos_indices_to_transfer = np.random.choice(train_pos_indices, size = len(val_pos_indices)//2)
    train_neg_indices_to_transfer = np.random.choice(train_neg_indices, size = len(val_neg_indices)//2)
    train_indices_to_transfer = np.concatenate((train_pos_indices_to_transfer, train_neg_indices_to_transfer))
    train_indices_not_transfer = [i for i in range(train_dataset.imgs.shape[0]) if i not in train_indices_to_transfer]

    train_dataset_v1 = copy.deepcopy(train_dataset)
    train_dataset_v1.imgs = train_dataset.imgs[train_indices_not_transfer,:,:]
    train_dataset_v1.labels = train_dataset.labels[train_indices_not_transfer]
    # train_dataset_v1.imgs = np.concatenate((train_dataset_v1.imgs, val_dataset.imgs[val_indices_to_transfer,:,:]), axis=0)
    val_dataset_v1 = copy.deepcopy(val_dataset)
    val_dataset_v1.imgs = np.concatenate((val_dataset_v1.imgs, train_dataset.imgs[train_indices_to_transfer,:,:]), axis=0)
    val_dataset_v1.labels = np.concatenate((val_dataset_v1.labels, train_dataset.labels[train_indices_to_transfer]), axis=0)

    return train_dataset_v1, val_dataset_v1



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

    train_transformation_list = []
    eval_transformation_list = []

    shape_transform = True

    if not args.data_aug:
        print("INFO: NOT Using Data Augmentation")
        if n_dim <= 3:
            train_transformation_list.extend([transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Grayscale(3),
                transforms.ToTensor()])
            eval_transformation_list.extend([transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Grayscale(3),
                transforms.ToTensor()])
            
            if resize:
                train_transformation_list.append(transforms.Resize((32, 32), interpolation=Image.NEAREST))
                eval_transformation_list.append(transforms.Resize((32, 32), interpolation=Image.NEAREST))

        train_transformation_list.extend([transforms.Resize((32, 32)), transforms.Normalize(mean=[.5], std=[.5])])
        eval_transformation_list.extend([transforms.Resize((32, 32)), transforms.Normalize(mean=[.5], std=[.5])])

    else:
        print("INFO : Using Data Augmentation")
        if n_dim <= 3:
            train_transformation_list.extend([transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Grayscale(3),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=30),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                # GaussianBlur(kernel_size, sigma=(0.1, 2.0)), 
                transforms.ToTensor()
            ])

            eval_transformation_list.extend([transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Grayscale(3),
                transforms.ToTensor()
            ])
        
            if resize:
                train_transformation_list.append(transforms.Resize((32, 32), interpolation=Image.NEAREST))
                eval_transformation_list.append(transforms.Resize((32, 32), interpolation=Image.NEAREST))

            train_transformation_list.extend([transforms.Resize((32, 32)), transforms.Normalize(mean=[.5], std=[.5])])
            eval_transformation_list.extend([transforms.Resize((32, 32)), transforms.Normalize(mean=[.5], std=[.5])])

        elif n_dim == 4:

            max_displacement = 15, 10, 0  # in x, y and z directions
            train_transform = [
                                # Transform3D(mul='random', flip=True) if shape_transform else Transform3D(), 
                                tio.RandomAffine(),  
                               tio.RandomFlip(axes=['inferior-superior'], flip_probability=0.5),
                               tio.RandomBlur()
                            #    tio.RandomNoise(std=0.5)
                            #    ,tio.RandomElasticDeformation(max_displacement=max_displacement)
                               ]
            # eval_transform = [Transform3D(mul='0.5') if shape_transform else Transform3D()]
            train_transformation_list.extend(train_transform)
            # eval_transformation_list.extend(eval_transform)

            if resize:
                train_transformation_list.append(transforms.Resize((32, 32), interpolation=Image.NEAREST))
                eval_transformation_list.append(transforms.Resize((32, 32), interpolation=Image.NEAREST))


    data_train_transform = transforms.Compose(train_transformation_list)
    data_eval_transform = transforms.Compose(eval_transformation_list)


    train_dataset = DataClass(split='train', transform=data_train_transform, download=download, as_rgb=as_rgb)
    # val_dataset = DataClass(split='val', transform=data_eval_transform, download=download, as_rgb=as_rgb)
    test_dataset = DataClass(split='test', transform=data_eval_transform, download=download, as_rgb=as_rgb)

    # if args.modify_datasets=='halve_val':
    #     print("INFO : Halving the validation dataset")
    #     train_dataset, val_dataset  = halve_val_dset(train_dataset, val_dataset)
    # elif args.modify_datasets=='double_val':
    #     print("INFO : Doubling the validation dataset")
    #     train_dataset, val_dataset  = double_val_dset(train_dataset, val_dataset)
    


    if n_dim == 4:
        # train_dataset.imgs = np.swapaxes(train_dataset.imgs, 1, 3) 
        # val_dataset.imgs = np.swapaxes(val_dataset.imgs, 1, 3) 
        test_dataset.imgs = np.swapaxes(test_dataset.imgs, 1, 3) 
        # train_dataset.imgs = torch.tensor(torch.from_numpy(train_dataset.imgs))
        train_dataset = MNIST_3D_Datasets(train_dataset, transformation_list=train_transformation_list, 
                                        as_rgb=train_dataset.as_rgb, target_transform=train_dataset.target_transform)
        # val_dataset = MNIST_3D_Datasets(val_dataset, transformation_list=eval_transformation_list, 
        #                                 as_rgb=train_dataset.as_rgb, target_transform=train_dataset.target_transform)
        test_dataset = MNIST_3D_Datasets(test_dataset, transformation_list=eval_transformation_list, 
                                        as_rgb=train_dataset.as_rgb, target_transform=train_dataset.target_transform)
    
    # sampler = DualSampler(train_dataset, batch_size, sampling_rate=sampling_rate)
    # sampler = ImbalancedDatasetSampler(train_dataset)

    
    # train_loader = data.DataLoader(dataset=train_dataset,
    #                             batch_size=batch_size,
    #                             # sampler=sampler,
    #                             shuffle=True)
    # train_loader_at_eval = data.DataLoader(dataset=train_dataset,
    #                             batch_size=batch_size,
    #                             shuffle=False)
    # val_loader = data.DataLoader(dataset=val_dataset,
    #                             batch_size=batch_size,
                                # shuffle=False)
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


    # train_evaluator = medmnist.Evaluator(data_flag, 'train')
    # val_evaluator = medmnist.Evaluator(data_flag, 'val')
    test_evaluator = medmnist.Evaluator(data_flag, 'test')

    if loss_fn == 'CE':
        if task == "multi-label, binary-class":
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
    elif loss_fn == 'AUCM':
        criterion = AUCMLoss()
    elif loss_fn == 'pAUC':
        beta = 0.1
        eta = 1e1
        criterion = pAUCLoss(pos_len=len(np.where(train_dataset.labels==1)[0]), backend='SOPA', beta=beta, margin=margin)
    print("Loss Function Used : {}".format(criterion))

    # IN CASE OF PRETRAINED
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device)['net'], strict=True)
        # train_metrics = test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run, output_root)
        # val_metrics = test(model, val_evaluator, val_loader, task, criterion, device, run, output_root)
        test_metrics = test(model, test_evaluator, test_loader, task, criterion, device, run, output_root)

        # print('train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2]) + \
        #       'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2]) + \
        print('test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2]))

    # if num_epochs == 0:
    #     return

    # if optimizer_fn == 'Adam':
    #     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # elif optimizer_fn == 'PESG':
    #     optimizer = PESG(model, 
    #              loss_fn=criterion, 
    #              lr=lr, 
    #              momentum=momentum, 
    #              margin=margin, 
    #              epoch_decay=epoch_decay, 
    #              weight_decay=weight_decay)
    # elif optimizer_fn=='SOPA':
    #     optimizer = SOPA(model.parameters(), loss_fn=criterion.loss_fn, mode='adam', lr=lr, eta=eta, weight_decay=weight_decay)
    # print("Optimizer Used : {}".format(optimizer.__str__()[:5]))

    # test_metrics = test(model, test_evaluator, test_loader, task, criterion, device, run, None,  loss_fn)
    # print("Test AUC : {} \t Test Accuracy : {}".format(test_metrics[1], test_metrics[2]) )


def test(model, evaluator, data_loader, task, criterion, device, run, save_folder=None, loss_fn=None):

    model.eval()
    
    total_loss = []
    y_score = torch.tensor([]).to(device)
    target_list = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):

            if len(data_loader.dataset.imgs.shape)==4:
                inputs = inputs.squeeze()
            try:
                try:
                    outputs = model(inputs.to(device))
                except:
                    # pdb.set_trace()
                    if  len(data_loader.dataset.imgs.shape) != inputs.ndim:
                        inputs = inputs.unsqueeze(0)
                    outputs = model(inputs.float().to(device))
            except:
                pdb.set_trace()
            
            # if loss_fn == 'CE':
            #     if task == 'multi-label, binary-class':
            #         targets = targets.to(torch.float32).to(device)
            #         loss = criterion(outputs, targets)
            #         m = nn.Sigmoid()
            #         outputs = m(outputs).to(device)
            #     else:
            #         targets = torch.squeeze(targets, 1).long().to(device)
            #         loss = criterion(outputs, targets)
            #         m = nn.Softmax(dim=1)
            #         outputs = m(outputs).to(device)
            #         targets = targets.float().resize_(len(targets), 1)
            # elif loss_fn == 'AUCM':
            targets = torch.squeeze(targets, 1).long().to(device)
            loss = criterion(outputs, targets)
            outputs = outputs.to(device)
                # targets = targets.float().resize_(len(targets), 1)

            total_loss.append(loss.item())
            y_score = torch.cat((y_score, outputs), 0)
            target_list = torch.cat((target_list, targets))

        y_score = y_score.detach().cpu().numpy()
        # auc, acc = evaluator.evaluate(y_score, save_folder=None, run=run)
        sklearn_auc = roc_auc_score(target_list.cpu(), [pred[1] for pred in y_score])
        sklearn_acc = acc_score(target_list.cpu(), y_score[:,-1]>0.5)
        auc, acc = sklearn_auc, sklearn_acc

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
                        default='AUCM',
                        help='Loss function to train the model',
                        choices=['AUCM', 'CE', 'pAUC'],
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
                        choices=['Adam','PESG', 'SOPA'],
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
                        default=1e-7,
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
    parser.add_argument('--data_aug',
                        action="store_true",
                        help='To perform data augmentation')
    parser.add_argument('--comment',
                        help='Comment for saving files',
                        type=str)  
    parser.add_argument('--mode',
                        choices=['test', 'train'],
                        help='Mode of running',
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