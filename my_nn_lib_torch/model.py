import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from torchvision import datasets, transforms
from tqdm import tqdm

from .core import BaseModule



# import mnist



class MyModel(BaseModule):
    def __init__(self, layer_list):
        self.layers = layer_list
        self.grad = 0.0

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)

        return X
    
    def backward(self, label):

        idx_maxlayer = len(self.layers)-1
        dLdZ = label

        for idx in range(idx_maxlayer, -1, -1):
            # 倒敘遍歷

            dLdZ = self.layers[idx].backward(delta=dLdZ)
            # if isinstance(dLdZ, tuple):
            #     dLdZ = dLdZ[0]
            #     self.grad = dLdZ[1]
            
    
    def update_params(self, opt_params):
        for layer in self.layers:
            layer.update_params(opt_params)
        
    def get_pred(self, X, with_onehot=False):
        pred = self.forward(X)
        if with_onehot:
            return pred
        return torch.argmax(pred, axis=1)

    def check_err(self, pbar, grad_arr, w_arr):
        grad_m = torch.mean(self.grad).cpu().detach().numpy()
        w_m = torch.mean(self.layers[1].w).cpu().detach().numpy()
        
        pbar.set_postfix({'grad': grad_m,
                          'w': w_m})
        
        if np.isnan(grad_m) or np.isnan(w_m):
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.plot(w_arr)
            plt.subplot(1, 2, 2)
            plt.plot(grad_arr)
            plt.show()
            return True
                    
        # time.sleep(0.1)
        grad_arr.append(grad_m)
        w_arr.append(w_m)
        return False
        
    
    def train_with_dataset(self, dataset, loss_func, hyper_params: dict, show_plot=False):
   
        
        train_loss_arr = []
        val_loss_arr = []
        
        val_acc_arr = []
     
        if isinstance(dataset, tuple):
            x_all, y_all = dataset[0], dataset[1]
            split_ratio = 0.8
            split_idx = int(len(x_all) * split_ratio)
            x_train = x_all[:split_idx]
            y_train = y_all[:split_idx]
            x_val = x_all[split_idx:]
            y_val = y_all[split_idx:]


            train_samples = x_train.shape[0]
            val_samples = x_val.shape[0]

            # 將 train data 打包成 batch
            X_batch_train, Y_batch_train = self.pack_to_batch(x_train, y_train, hyper_params['batch_size'], train_samples)
            X_batch_val, Y_batch_val = self.pack_to_batch(x_val, y_val, hyper_params['batch_size'], val_samples)

            train_loader = list(zip(X_batch_train, Y_batch_train))
            val_loader = list(zip(X_batch_val, Y_batch_val))


        # split train and val
        else:
            split_ratio = 0.8
            split_idx = int(len(dataset) * split_ratio)
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [split_idx, len(dataset) - split_idx])


            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=hyper_params['batch_size'], shuffle=True)
            val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=hyper_params['batch_size'], shuffle=False)

            train_samples = train_loader.dataset.dataset.data.shape[0]
            val_samples = val_loader.dataset.dataset.data.shape[0]

        

    
        for i in range(hyper_params['epoch']):
            train_loss = 0
            val_loss = 0
            val_acc = 0

            grad_mean_arr = []
            w_mean_arr = []
        
            with tqdm(total=len(train_loader), desc=f'Epoch {i} train') as pbar:
                for idx, (X_batch, Y_batch) in enumerate(train_loader):
                    # 單個 batch 訓練過程
                    # 1. 前向傳播
                    # 2. 反向傳播
                    # 3. 更新權重   
                    
                    Y_batch = self.one_hot_encoding(Y_batch, 10)
                    X_batch, Y_batch = X_batch.cuda(), Y_batch.cuda()
                    
                    self.forward(X_batch)
                    self.backward(Y_batch)
                    self.update_params({'lr': hyper_params['lr'], 'alpha': hyper_params['alpha']})
                    train_loss += loss_func.cal_loss(self.get_pred(X_batch, with_onehot=True), Y_batch)

                    # if self.check_err(pbar, grad_mean_arr, w_mean_arr): return 0
                    pbar.update(1)
            train_loss = train_loss.cpu().numpy() / train_samples
            train_loss_arr.append(train_loss)

            print('train Loss:', round(train_loss, 4))

            with tqdm(total=len(val_loader), desc='val') as pbar:
                for idx, (X_batch, Y_batch) in enumerate(val_loader):
    
                    Y_batch = self.one_hot_encoding(Y_batch, 10)
                    X_batch, Y_batch = X_batch.cuda(), Y_batch.cuda()
                    
                    predictions = self.get_pred(X_batch, with_onehot=True)
                    val_loss += loss_func.cal_loss(predictions, Y_batch)
                    val_acc += torch.round(self.calculate_acc(predictions, Y_batch), decimals=2)
                    pbar.update(1)

            val_loss = val_loss.cpu().numpy() / val_samples
            val_acc = val_acc.cpu().numpy()  / len(val_loader)

            
            val_loss_arr.append(val_loss)

            

            val_acc_arr.append(val_acc)
            print('val Loss:', round(val_loss, 4))
            print(f'Val Acc:{round(val_acc, 4) * 100} %')

        if show_plot:
            self.plot_loss_acc(train_loss_arr, val_loss_arr, val_acc_arr)

        return train_loss_arr, val_loss_arr, val_acc_arr
    
    def one_hot_encoding(self, Y, n_classes):
        # 將 Y 轉換成 one-hot encoding
        if len(Y.shape) == 2:
            Y_onehot = torch.zeros((len(Y), n_classes))
            Y_onehot[torch.arange(len(Y), device=Y.device).long(), Y.long().squeeze(1)] = 1
        else:
            Y_onehot = torch.zeros((len(Y), n_classes))
            Y_onehot[torch.arange(len(Y), device=Y.device).long(), Y.long()] = 1
        return Y_onehot

    def calculate_acc(self, predictions, Y):
        Y = torch.argmax(Y, axis=1)
        predictions = torch.argmax(predictions, axis=1)
        return torch.sum(predictions == Y) / len(Y)
    
    def pack_to_batch(self, X, Y, bs, n_samples):

        # 將全部的資料打包成 batch，每個 batch 的大小為 bs
        # 若 n_samples 不能被 bs 整除，則將 X_train, Y_all 進行 padding
        n_dim = len(X.shape)
        if X.shape[0] % bs != 0:  
            pad_idx = [0] * 2 * n_dim
            pad_idx[-1] = bs - (n_samples % bs)
            X = F.pad(X, pad_idx)
            Y = F.pad(Y, (0, bs - (n_samples % bs)))

            # X = np.pad(X, [[0, bs - (n_samples % bs)] if i == 0 else [0, 0] for i in range(n_dim)], 'constant', constant_values=(0))
            # Y = np.pad(Y, [[0, bs - (n_samples % bs)], [0, 0]], 'constant', constant_values=(0))

        if n_dim == 2:
            X_batch_all = X.reshape(-1, bs, X.shape[1])
            # Y is a value here, not one-hot
            Y_batch_all = Y.reshape(-1, bs, 1)
        elif n_dim == 4:
            X_batch_all = X.reshape(-1, bs, X.shape[1], X.shape[2], X.shape[3])
            # Y is a value here, not one-hot
            Y_batch_all = Y.reshape(-1, bs, 1)
        else:
            raise NotImplementedError

        # 從最後一個 batch 拿掉 padding 的部分
        if X.shape[0] % bs != 0:
            X_batch_all[-1] = X_batch_all[-1][:(n_samples % bs)]
            Y_batch_all[-1] = Y_batch_all[-1][:(n_samples % bs)]

        # X_batch_all -> (n_batch, batch_size, n_features)
        # Y_batch_all -> (n_batch, batch_size, n_classes)
        return X_batch_all, Y_batch_all
    
    def plot_loss_acc(self, train_loss_arr, val_loss_arr, val_acc_arr):
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 2, 1)
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(train_loss_arr)
        plt.plot(val_loss_arr)
        plt.legend(['Train Loss', 'Val Loss'])

        plt.subplot(1, 2, 2)
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel('Val Acc')
        plt.plot(val_acc_arr)
        plt.show()