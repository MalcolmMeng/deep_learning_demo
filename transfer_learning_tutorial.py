# -*- coding: utf-8 -*-

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import json
from sklearn.metrics import confusion_matrix

# Data augmentation and normalization for training

# Just normalization for validation
data_mean_bak=[0.485, 0.456, 0.406]
data_std_bak=[0.229, 0.224, 0.225]
data_mean=[0.616,0.589,0.584]
data_std=[0.325,0.324,0.322]

data_dir = ''
image_datasets = {}
dataloaders = {}
dataset_sizes = {}
class_names = []
data_transforms={}
device=''
hyper_batch_size=''
def init_data():
    global data_dir,image_datasets,dataloaders,dataset_sizes,class_names,device
    data_dir = r'/home/malcolm/data/home/Untitled Folder'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val','test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                                  shuffle=True, num_workers=8)
                   for x in ['train', 'val','test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}
    class_names = image_datasets['train'].classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def model_factorty(name,output_num,test=False):
    model=models.resnet18()
    if name=='resnet18':
        model=models.resnet18(pretrained=True)
    elif name=='resnet50':
        model=models.resnet50(pretrained=True)
    elif name=='inception_v3':
        model=models.inception_v3(pretrained=True)
    num_ftrs=model.fc.in_features
    model.fc=nn.Linear(num_ftrs,output_num)
    if test:
        model.load_state_dict(torch.load('parameters_'+name+'.pkl'))
    model=model.to(device)
    return model


def data_transforms_factory(net_name,data_mean,data_std):
    tmp_size=256
    final_size=224
    if net_name=='inception_v3':
        tmp_size=360
        final_size=299
    data_transforms = {
        'train': transforms.Compose([
            # 224 ->299
            transforms.RandomResizedCrop(final_size,(0.7,1.0)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),  # flip within [-10 degreee, +10 degree]
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),  # brightness, contrast, saturation, hue
            #TOTensor postion
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std)
        ]),
        'val': transforms.Compose([
            # (256,224)->(360,299)
            transforms.Resize(tmp_size),
            transforms.CenterCrop(final_size),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(tmp_size),
            transforms.CenterCrop(final_size),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std)
        ]),
    }
    return data_transforms



info_dict={}
def info_json(**kvargs):
    ''';
    net_name:string
    phase :string including train val test
    '''
    if kvargs.get('net_name')==None:
        raise  ValueError('lack the parameters : net_name')
    if info_dict.get('net_name')==None or kvargs.get('net_name')!=info_dict.get('net_name'):
        info_dict.clear()
        info_dict['net_name']=kvargs['net_name']
    if kvargs['phase'] != 'test':
        phase,loss,err=kvargs['phase'],kvargs['loss'],kvargs['err']
        info_dict[phase]={'loss':loss,'err':err}
    else:
        phase,confu_mat,excu_time,acc=kvargs['phase'],\
        kvargs['confu_mat'],kvargs['excu_time'],kvargs['acc']
        info_dict[phase]={'confu_mat':confu_mat,'excu_time':excu_time,'acc':acc}


def train_model(net_name,model,criterion, optimizer, scheduler, num_epochs=25):
    # model = model_factorty(net_name, output_num)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    trace_loss={'train':[],'val':[]}
    trace_err={'train':[],'val':[]}
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #outputs=model(inputs)
                    if phase=='train':
                        if net_name=='inception_v3':
                            outputs,aux= model(inputs)
                        else:
                            outputs=model(inputs)
                    else:
                        outputs=model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = float(running_loss / dataset_sizes[phase])
            epoch_err = 1-float(running_corrects.double() / dataset_sizes[phase])
            trace_loss[phase].append(epoch_loss)
            trace_err[phase].append(epoch_err)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, 1-epoch_err))

            # deep copy the model
            if phase == 'val' and (1-epoch_err) > best_acc:
                best_acc = 1-epoch_err
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
    time_elapsed = time.time() - since
    info_json(net_name=net_name,phase='train',loss=trace_loss['train'],err=trace_err['train'])
    info_json(net_name=net_name,phase='val',loss=trace_loss['val'],err=trace_err['val'])
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'parameters_' + net_name + '.pkl')
    return model



def test_model(net_name,model):
    print('-' * 10)
    print('start test...')
    model.eval()
    st=time.time()
    run_corrects=0
    y_true=[]
    y_pred=[]
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        run_corrects += torch.sum(preds == labels.data)
        preds=torch.Tensor.cpu(preds).numpy()
        labels=torch.Tensor.cpu(labels).numpy()
        y_pred+=list(preds)
        y_true+=list(labels)
    st=time.time()-st
    print(class_names)
    print('the confusion_matrix is')
    mat=confusion_matrix(y_true,y_pred)
    mat=mat/mat.sum(axis=1)[:,None]
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    print(mat)
    np.set_printoptions()
    print("total excute time {:.0f}m{:.0f}s".format(st//60,st%60))
    print("right num {}".format(run_corrects))
    print("accuracy is {:.4f}".format(run_corrects.double() / dataset_sizes['test']))
    info_json(net_name=net_name,phase='test',confu_mat=mat.tolist(),excu_time=st,acc=float(run_corrects.double() / dataset_sizes['test']))
if __name__=='__main__':
    net_name = 'resnet50'
    print('this is '+net_name+'\n'+10*'#')
    output_num = 5
    test=False
    data_transforms = data_transforms_factory(net_name, data_mean, data_std)
    init_data()
    if not test:
        criterion = nn.CrossEntropyLoss()
        model_ft=model_factorty(net_name,output_num)
        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        train_model(net_name,model_ft,criterion,optimizer_ft,exp_lr_scheduler)
        test_model(net_name,model_ft)
    else:
        model_tt=model_factorty(net_name,output_num,test=True)
        test_model(net_name,model_tt)
    print(info_dict)
    with open(net_name+'detail.json','w') as f:
        json.dump(info_dict,f)
