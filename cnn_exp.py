import torch
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import random
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

from torchvision import models
# from torchsummary import summary
import csv

import sys
import os

# =============================================================================
# 1. Hyperparameter
# =============================================================================

# Choose GPU
device = torch.device('cuda:1')

# Clip Image
H_min, H_max = 300, 600 # the height range of an img 
W_min, W_max = 300, 600 # the width range of an img 

# Smooth Pixel
smooth = 2
H = int((H_max - H_min) / smooth) # num of height pixels
W = int((W_max - W_min) / smooth) # num of width pixels

# timestep
total_forward = 4 * 4
data_len = 4 * 26

# experiment number
sample_num = 20

# starting probability
starting_prob = 'original'

# noise of starting probability
is_noise_sp = False
noise_name = 'no-noise'
sp_noise_ratio = 0
# noise_type = 'normal'
noise_type = 'uniform'

# train label mutation
is_mutate = False
is_only_0 = False
pos_ratio = 0.05

#mutation choice
mutation_ratio = float(sys.argv[1])

exp_type = sys.argv[2]
cube_shape = int(sys.argv[3])

if exp_type == 'general':
    
    to_average_cube_mutation = False
    average_to_one = False
    to_cube_mutation = False
    to_multinominal_cube_mutation = False
    to_general_mutation = True
    to_only_0_mutation = False
    
elif exp_type == 'cube':
    
    to_average_cube_mutation = False
    average_to_one = False
    to_cube_mutation = True
    to_multinominal_cube_mutation = False
    to_general_mutation = False
    to_only_0_mutation = False

elif exp_type == 'cube_to_one':
    
    to_average_cube_mutation = True
    average_to_one = True
    to_cube_mutation = False
    to_multinominal_cube_mutation = False
    to_general_mutation = False
    to_only_0_mutation = False
    
elif exp_type == 'cube_multi':
    
    to_average_cube_mutation = False
    average_to_one = False
    to_cube_mutation = False
    to_multinominal_cube_mutation = True
    to_general_mutation = False
    to_only_0_mutation = False

elif exp_type == 'no_mutation':
    
    to_average_cube_mutation = False
    average_to_one = False
    to_cube_mutation = False
    to_multinominal_cube_mutation = False
    to_general_mutation = False
    to_only_0_mutation = False
    
else:
    print('========== Wrong parameters: {}, {}, {} =========='.format(mutation_ratio, exp_type, cube_shape))
    sys.exit(0)

# cube/mutation choice
if to_average_cube_mutation and average_to_one:
    mutation = 'avg-cube-to-one'
    pos_ratio = 0
elif to_cube_mutation:
    mutation = 'cube'
    pos_ratio = mutation_ratio
elif to_multinominal_cube_mutation:
    mutation = 'cube-multinominal'
    pos_ratio = mutation_ratio
elif to_general_mutation:
    mutation = 'general'
    pos_ratio = mutation_ratio
    cube_shape = 1
else:
    mutation = 'no-mutation'
    pos_ratio = 0
    cube_shape = 1

# save as
if mutation == 'no-mutation':
    file_title_1 = '{}%_direct_cube-shape_{}_{}_times_test'.format(pos_ratio*100, cube_shape, sample_num)
    file_title_2 = '{}%_{}_cube-shape_{}_{}_times_test'.format(pos_ratio*100, mutation, cube_shape, sample_num)
    print(file_title_1)
    print(file_title_2)
else:
    file_title = '{}%_{}_cube-shape_{}_{}_times_test'.format(pos_ratio*100, mutation, cube_shape, sample_num)
    print(file_title)

folder_name = 'sep_30_days_x_361_y_373'
# folder_name = 'jun_30_days_x_525_y_000_label_2'

# Validation data
validation_folder_name = 'aug_30_days_x_361_y_373'
# validation_folder_name = 'may_30_days_x_525_y_000_label_2'

# =============================================================================
# 2. Read Data
# =============================================================================

train_label = np.load('../data_preprocess/{}/train_label_smooth_{}.npy'.format(folder_name, smooth))
test_label = np.load('../data_preprocess/{}/test_label_smooth_{}.npy'.format(folder_name, smooth))
train_img = np.load('../data_preprocess/{}/train_img_smooth_{}.npy'.format(folder_name, smooth))
test_img = np.load('../data_preprocess/{}/test_img_smooth_{}.npy'.format(folder_name, smooth))

# For cube mutation
train_label_mutation = train_label

train_label = torch.tensor(train_label, dtype=torch.float32)
test_label = torch.tensor(test_label, dtype=torch.float32)

# =============================================================================
# 3. Model
# =============================================================================

class DirectForwardModel(torch.nn.Module):
    def __init__(self):
        super(DirectForwardModel, self).__init__()
        # encoder
        self.batch = torch.nn.BatchNorm2d(16)
        self.pad = torch.nn.ReplicationPad2d(1)
        self.cnn = torch.nn.Conv2d(16, 32, kernel_size=3)
        
        # decoder
        self.linear = torch.nn.Linear(32, total_forward)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # input shape of x = (N, C_in, H, W)
        x = self.cnn(self.pad(self.batch(x))) # shape of x = (N, C_out, H, W)
        x = x.transpose(1,3) # shape of x = (N, W, H, C_out)
        x = self.sigmoid(self.linear(x)).transpose(1,3) # shape = (N, F, H, W)
        
        return x

class ConditionalForwardModel(torch.nn.Module):
    def __init__(self):
        super(ConditionalForwardModel, self).__init__()
        # encoder
        self.batch = torch.nn.BatchNorm2d(16)
        self.pad = torch.nn.ReplicationPad2d(1)
        self.cnn = torch.nn.Conv2d(16, 32, kernel_size=3)
        
        # decoder
        self.linearT = torch.nn.Linear(32, total_forward)
        self.linearF = torch.nn.Linear(32, total_forward)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # input shape of x = (N, C_in, H, W)
        x = self.cnn(self.pad(self.batch(x))) # shape of x = (N, C_out, H, W)
        x = x.transpose(1,3) # shape of x = (N, W, H, C_out)
        
        x_T = self.sigmoid(self.linearT(x)).transpose(1,3)
        x_F = self.sigmoid(self.linearF(x)).transpose(1,3)
        
        return x_T, x_F # shape = (N, F, H, W)

    def predict(self, x, init_cond):
        # input shape of x = (N, C_in, H, W)
        # input shape of init_cond = (N, H, W)
        x = self.cnn(self.pad(self.batch(x))) # shape of x = (N, C_out, H, W)
        x = x.transpose(1,3) # shape of x = (N, W, H, C_out)
        
        x_T = self.sigmoid(self.linearT(x)).transpose(1,3) # shape = (N, F, H, W)
        x_F = self.sigmoid(self.linearF(x)).transpose(1,3)
        
        x_marg = [x_T[:,0] * init_cond + x_F[:,0] * (1 - init_cond)] # shape = (N, H, W)
        for i in range(1, x_T.shape[1]):
            x_marg.append(x_T[:,i] * x_marg[-1] + x_F[:,i] * (1 - x_marg[-1]))
        x_marg = torch.stack(x_marg, dim=1) # shape = (N, F, H, W)
        
        return x_marg
    
# For starting probability
def gen_pred_label(label_list, data_len): # label_list = list[np[array(40, 150, 150)]]
    
    pred_label = [label_list[:data_len].astype(np.float32)]
    np_pred_label = np.asarray(pred_label, dtype=np.float32)
    
    total_pixels = np_pred_label.shape[-1] * np_pred_label.shape[-2]
    horizon_sum = np.sum(np.sum(np_pred_label, axis=-1), axis=-1)
    starting_probability = np.divide(horizon_sum, total_pixels)
    pred_shape = np_pred_label.shape
    new_pred_label = np.ones(pred_shape, dtype=np.float32)
    
    for i in range(pred_shape[1]):
        new_pred_label[0, i] = new_pred_label[0, i] * starting_probability[0, i]
    
    pred_label = [new_pred_label[0]]
    
    return pred_label

def cube_mutation(train_label_mutation,mutation_ratio,cube_shape): # this can only be used on three dimensional label,cube size = (2*cube_shape+1)^3
    # choose cube mutation label
    train_label_mutation = train_label_mutation
    for i in range(len(train_label_mutation)):
        for j in range(len(train_label_mutation[0])):
            for k in range(len(train_label_mutation[0][0])):
                #if nearby contents includes 1, then add mutation
                check_cube = train_label_mutation[max(0, i-cube_shape):min(train_label_mutation.shape[0],i + cube_shape + 1),max(0,j-cube_shape):min(train_label_mutation.shape[1],j + cube_shape + 1),max(0,k-cube_shape):min(train_label_mutation.shape[2],k + cube_shape + 1)]
                if (random.random() < mutation_ratio) and (train_label_mutation[i][j][k] == 0) and (1 in np.array(check_cube).flatten()):
                    train_label_mutation[i][j][k] = 2 # if element is 2, then it will be mutated later 
    
    # start mutation
    for i in range(len(train_label_mutation)):
        for j in range(len(train_label_mutation[0])):
            for k in range(len(train_label_mutation[0][0])):
                if train_label_mutation[i][j][k] == 2:
                    train_label_mutation[i][j][k] = 1
    return train_label_mutation

# 記得把label傳進來以前要設成astype(float)，不然np.array預設只能存整數。
# train_label_mutation = train_label_mutation.astype(float)
def average_cube_mutation(train_label_mutation,cube_shape,to_one): # this can only be used on three dimensional label,cube size = (2*cube_shape+1)^3, to one => 用平均機率換成1
    # choose cube mutation label
    change_prob = {}
    train_label_mutation = train_label_mutation.astype(float)
    for i in range(len(train_label_mutation)):
        for j in range(len(train_label_mutation[0])):
            for k in range(len(train_label_mutation[0][0])):
                #if nearby contents includes 1, then add mutation. Check_cube = nearby contents
                check_cube = train_label_mutation[max(0, i-cube_shape):min(train_label_mutation.shape[0],i + cube_shape + 1),max(0,j-cube_shape):min(train_label_mutation.shape[1],j + cube_shape + 1),max(0,k-cube_shape):min(train_label_mutation.shape[2],k + cube_shape + 1)]
                # if given label = 0, there is 1 in check_cube, there is mutation ratio% chance that this label will be switched
                if (train_label_mutation[i][j][k] == 0) and (1 in np.array(check_cube).flatten()):
                    train_label_mutation[i][j][k] = 2 # if element is 2, then it will be mutated later. I don't switch it directly because it might affect other labels later
                    change_prob[i,j,k] = np.count_nonzero(check_cube == 1.0)/(len(np.array(check_cube).flatten())-1)    
    # start mutation
    for i in range(len(train_label_mutation)):
        for j in range(len(train_label_mutation[0])):
            for k in range(len(train_label_mutation[0][0])):
                if train_label_mutation[i][j][k] == 2.0:
                    # change to 1
                    if to_one == True :
                        if random.random()<change_prob[(i,j,k)]:
                            train_label_mutation[i][j][k] = 1.0
                        else:
                            train_label_mutation[i][j][k] = 0.0
                    # use probability
                    elif to_one == False:
                        train_label_mutation[i][j][k] = change_prob[(i,j,k)]
    return(train_label_mutation)

# this can only be used on three dimensional label,cube size = (2*cube_shape+1)^3
# sample ratio determined the sample # in multinominal sampling. sample number = total candidate label * sample_ratio
def multinomial_cube_mutation(train_label_mutation,sample_ratio,cube_shape):  

    change_prob = {}
    train_label_mutation = train_label_mutation.astype(float) ##make a copy
    total_candidate = 0
    
    # start recording # of label = 1 in each cube
    for i in range(len(train_label_mutation)):
        for j in range(len(train_label_mutation[0])):
            for k in range(len(train_label_mutation[0][0])):
                # if nearby contents includes 1, then add mutation. Check_cube = nearby contents
                check_cube = train_label_mutation[max(0, i-cube_shape):min(train_label_mutation.shape[0],i + cube_shape + 1),max(0,j-cube_shape):min(train_label_mutation.shape[1],j + cube_shape + 1),max(0,k-cube_shape):min(train_label_mutation.shape[2],k + cube_shape + 1)]
                # if given label = 0, record its number of cube neighbor whose label = 1. If given label = 1 or its neighbor does not include 1, store 0
                if (train_label_mutation[i][j][k] == 0) and (1 in np.array(check_cube).flatten()):
                    change_prob[i,j,k] = np.count_nonzero(check_cube == 1.0)
                    total_candidate = total_candidate + 1
                else :
                    change_prob[i,j,k] = 0.0
    
    # count probability of multinominal data
    total = sum(change_prob.values())
    change_prob = {k:v/total for k,v in change_prob.items()}
    key = []
    value = []
    for k,v in change_prob.items():
        key.append(k)
        value.append(v)
    
    # sampling from multinomial distribution
    print(total_candidate*sample_ratio)
    sample = np.random.multinomial(int(total_candidate*sample_ratio), value) 
    
    # if a label is sampled, change its value to 1
    for i in range(len(key)):
        if sample[i] > 0:
            train_label_mutation[key[i][0],key[i][1],key[i][2]] = 1.0
            
    return(train_label_mutation)


def general_mutation(train_label_mutation,mutation_ratio):
    train_label_mutation = train_label_mutation.astype(float)
    for i in range(len(train_label_mutation)):
        for j in range(len(train_label_mutation[0])):
            for k in range(len(train_label_mutation[0][0])):
                if random.random() < mutation_ratio : 
                    if train_label_mutation[i][j][k] == 0:
                        train_label_mutation[i][j][k] = 1
                    else:
                        train_label_mutation[i][j][k] = 0
    return train_label_mutation

# =============================================================================
# 4. Data Preprocessing
# =============================================================================

if to_cube_mutation == True:
    train_label_mutation = cube_mutation(train_label_mutation, mutation_ratio, cube_shape = cube_shape)
if to_average_cube_mutation == True:
    train_label_mutation = average_cube_mutation(train_label_mutation, cube_shape = cube_shape, to_one = average_to_one)
if to_multinominal_cube_mutation == True:
    train_label_mutation = multinomial_cube_mutation(train_label_mutation, sample_ratio = mutation_ratio, cube_shape = cube_shape)
if to_general_mutation == True:
    train_label_mutation = general_mutation(train_label_mutation, mutation_ratio)
if to_only_0_mutation == True:
    train_label_mutation = only_0_mutation(train_label_mutation, mutation_ratio)


pred_train = gen_pred_label(train_label_mutation, data_len)
train_label_mutation = torch.tensor(train_label_mutation, dtype=torch.float32)

X_train = np.transpose(train_img[:data_len], [0,3,1,2])
X_train = torch.tensor(X_train) # shape = (N, C_in, H, W) = 464 horizons, 16 channels and 150*150image
Y_train = [train_label_mutation[i+1:i+total_forward+1] for i in range(data_len)]
Y_train = torch.stack(Y_train, dim=0) # shape = (N, F, H, W)= 464 horizons, 150 * 150 image, and 16 forward spans
cond_train = [train_label_mutation[i:i+total_forward] for i in range(data_len)] # get the exact label of a given x
cond_train = torch.stack(cond_train, dim=0) # shape = (N, F, H, W)
#init_cond_train = train_label_mutation[:data_len].to(device) # shape = (N, H, W)
if starting_prob == "average":
    init_cond_train = torch.tensor(pred_train[0]).to(device) # average probability as prior probability
elif starting_prob == "original":
    init_cond_train = train_label_mutation[:data_len].to(device) # shape = (N, H, W)


X_test = np.transpose(test_img[:data_len], [0,3,1,2])
X_test = torch.tensor(X_test) # shape = (N, C_in, H, W)
Y_test = [test_label[i+1:i+total_forward+1] for i in range(data_len)]
Y_test = torch.stack(Y_test, dim=0) # shape = (N, F, H, W)
#cond_test = [test_label[i:i+total_forward] for i in range(data_len)]
#cond_test = torch.stack(cond_test, dim=0) # shape = (N, F, H, W)
init_cond_test = test_label[:data_len].to(device) # shape = (N, H, W)
#init_cond_test = torch.tensor(pred_test[0]).to(device) # average probability as prior probability


dataset_train = TensorDataset(X_train, Y_train, cond_train)
dataset_test = TensorDataset(X_test, Y_test)

loader_train = DataLoader(dataset_train, shuffle=True, batch_size=1)
loader_test = DataLoader(dataset_test, shuffle=True, batch_size=1)

# =============================================================================
# 5. Train
# =============================================================================

# Multiple experiment

train_cond_auc = []
train_direct_auc = []
test_cond_auc = []
test_direct_auc = []


for log_num in range(sample_num):
    model_condi = ConditionalForwardModel().to(device)
    model_direc = DirectForwardModel().to(device)

    BCE = torch.nn.BCELoss()

    optim_condi = torch.optim.Adam(model_condi.parameters(), lr=5e-3)
    optim_direc = torch.optim.Adam(model_direc.parameters(), lr=5e-3)

    log = {'train': [], 'test': []}

    epochs = 50
    for epoch in range(epochs):
        print('\nEpoch:', epoch)
        log_train = {}
        log_test = {}
        log_train['epoch'] = epoch
        log_test['epoch'] = epoch
        total_loss_direc, total_loss_condi = 0, 0

        tbar = tqdm(enumerate(loader_train), total=data_len)
        for i, (x, y, cond) in tbar:
            # data to cuda
            x, y, cond = x.to(device), y.to(device), cond.to(device)

            # direct forward model
            output = model_direc(x)
            
            loss_direc = BCE(output, y)
            total_loss_direc += loss_direc.item()

            loss_direc.backward()
            optim_direc.step()
            optim_direc.zero_grad()

            # conditional forward model
            x_T, x_F = model_condi(x)

            mask_T = (cond == 1)
            mask_F = (cond == 0)

            loss_T = BCE(x_T[mask_T], y[mask_T])
            loss_T = 0 if torch.isnan(loss_T) else loss_T # rare event
            loss_F = BCE(x_F[mask_F], y[mask_F])

            loss_condi = loss_T + loss_F
            total_loss_condi += loss_condi.item()

            loss_condi.backward()
            optim_condi.step()
            optim_condi.zero_grad()

            # show
            tbar.set_postfix({'loss_direc': total_loss_direc / (i + 1), 
                              'loss_condi': total_loss_condi / (i + 1)})

        log_train['loss_direc'] = total_loss_direc / data_len
        log_train['loss_condi'] = total_loss_condi / data_len

        if (epoch+1) % 50 == 0:
            # train evaluate
            with torch.no_grad():
                print('\nEvaluate Train data:')

                # direct forward model
                output = model_direc(X_train.to(device)).detach().cpu() # shape = (N, F, W, H)

                print('compute auc_score...')
                log_train['auc_direc'] = []
                for i in range(total_forward):
                    score = roc_auc_score(Y_train[:,i].contiguous().view(-1).numpy(), 
                                          output[:,i].contiguous().view(-1).numpy())
                    log_train['auc_direc'].append(score)

                print('AUC of model_direc:')
                print(log_train['auc_direc'])

                # conditional forward model
                output = model_condi.predict(X_train.to(device), init_cond_train).detach().cpu() # shape = (N, F, W, H)

                print('compute auc_score...')
                log_train['auc_condi'] = []
                for i in range(total_forward):
                    score = roc_auc_score(Y_train[:,i].contiguous().view(-1).numpy(), 
                                          output[:,i].contiguous().view(-1).numpy())
                    log_train['auc_condi'].append(score)

                print('AUC of model_condi:')
                print(log_train['auc_condi'])        

            # test evaluate
            with torch.no_grad():
                print('\nEvaluate Test data:')

                # direct forward model
                output = model_direc(X_test.to(device)).detach().cpu() # shape = (N, F, W, H)

                print('compute auc_score...')
                log_test['auc_direc'] = []
                for i in range(total_forward):
                    score = roc_auc_score(Y_test[:,i].contiguous().view(-1).numpy(), 
                                          output[:,i].contiguous().view(-1).numpy())
                    log_test['auc_direc'].append(score)

                print('AUC of model_direc:')
                print(log_test['auc_direc'])

                # conditional forward model
                output = model_condi.predict(X_test.to(device), init_cond_test).detach().cpu() # shape = (N, F, W, H)

                print('compute auc_score...')
                log_test['auc_condi'] = []
                for i in range(total_forward):
                    score = roc_auc_score(Y_test[:,i].contiguous().view(-1).numpy(), 
                                          output[:,i].contiguous().view(-1).numpy())
                    log_test['auc_condi'].append(score)

                print('AUC of model_condi:')
                print(log_test['auc_condi'])  

            # logging
            log['train'].append(log_train)
            log['test'].append(log_test)
    
    # =============================================================================
    # 6. save
    # =============================================================================

    # log folder choosing
    output_folder = ''

    if folder_name == 'sep_30_days_x_361_y_373':
        output_folder = 'label_1'
    elif folder_name == 'jun_30_days_x_525_y_000_label_2':
        output_folder = 'label_2'


    # cube/mutation choice
    if to_average_cube_mutation and average_to_one:
        mutation = 'avg-cube-to-one'
        pos_ratio = 0
    elif to_cube_mutation:
        mutation = 'cube'
        pos_ratio = mutation_ratio
    elif to_multinominal_cube_mutation:
        mutation = 'cube-multinominal'
        pos_ratio = mutation_ratio
    elif to_general_mutation:
        mutation = 'general'
        pos_ratio = mutation_ratio
        cube_shape = 1
    else:
        mutation = 'no-mutation'
        pos_ratio = 0
        cube_shape = 1

    # save as
    if mutation == 'no-mutation':
        file_title_1 = '{}%_direct_cube-shape_{}_{}_times_test'.format(pos_ratio*100, cube_shape, sample_num)
        file_title_2 = '{}%_{}_cube-shape_{}_{}_times_test'.format(pos_ratio*100, mutation, cube_shape, sample_num)
        print(file_title_1)
        print(file_title_2)
        
        # save direct model
        os.makedirs(os.path.dirname('../src/exp_cnn_20/{}/{}/'.format(output_folder, file_title_1)), exist_ok=True)
        with open('../src/exp_cnn_20/{}/{}/log{}.csv'.format(output_folder, file_title_1, log_num),'w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Mutation_Type', 'S.P.', 'Train_test',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])

            write_train = [str(pos_ratio*100) + '%_direct'] + [starting_prob] + ['test'] + log_train['auc_direc']
            write_test = [str(pos_ratio*100) + '%_direct'] + [starting_prob] + ['test'] + log_test['auc_direc']

            writer.writerow(write_train)
            writer.writerow(write_test)

        # save conditional model
        os.makedirs(os.path.dirname('../src/exp_cnn_20/{}/{}/'.format(output_folder, file_title_2)), exist_ok=True)
        with open('../src/exp_cnn_20/{}/{}/log{}.csv'.format(output_folder, file_title_2, log_num),'w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Mutation_Type', 'S.P.', 'Train_test',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])

            write_train = [str(pos_ratio*100) + '%_' + mutation] + [starting_prob] + ['test'] + log_train['auc_condi']
            write_test = [str(pos_ratio*100) + '%_' + mutation] + [starting_prob] + ['test'] + log_test['auc_condi']

            writer.writerow(write_train)
            writer.writerow(write_test)
  
    else:
        file_title = '{}%_{}_cube-shape_{}_{}_times_test'.format(pos_ratio*100, mutation, cube_shape, sample_num)
        print(file_title)

        os.makedirs(os.path.dirname('../src/exp_cnn_20/{}/{}/'.format(output_folder, file_title)), exist_ok=True)
        with open('../src/exp_cnn_20/{}/{}/log{}.csv'.format(output_folder, file_title, log_num),'w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Mutation_Type', 'S.P.', 'Train_test',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])

            write_train = [str(pos_ratio*100) + '%_' + mutation] + [starting_prob] + ['test'] + log_train['auc_condi']
            write_test = [str(pos_ratio*100) + '%_' + mutation] + [starting_prob] + ['test'] + log_test['auc_condi']

            writer.writerow(write_train)
            writer.writerow(write_test)
    
