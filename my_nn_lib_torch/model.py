import numpy as np
from tqdm import tqdm
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


from .core import BaseModule



# import mnist



class MyModel(BaseModule):
    def __init__(self, layer_list):
        self.layers = layer_list

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)

        return X
    
    def backward(self, X, label):

        idx_maxlayer = len(self.layers)-1
        dLdZ = label

        for idx in range(idx_maxlayer, -1, -1):
            # 倒敘遍歷

            dLdZ = self.layers[idx].backward(delta=dLdZ)
            
    
    def update_params(self, opt_params):
        for layer in self.layers:
            layer.update_params(opt_params)
        
    def get_pred(self, X, with_onehot=False):
        pred = self.forward(X)
        if with_onehot:
            return pred
        return torch.argmax(pred, axis=1)

    def train(self, X_train, Y_train, X_val, Y_val, loss_func, hyper_params: dict, show_plot=False):
        # X_train -> (n_samples, n_features)
        # Y_train -> (n_samples, n_classes) one-hot 
        
        # params = self.weight_init(self.params_set_list)

        n_samples = X_train.shape[0]

        # 將 train data 打包成 batch
        X_batch_all, Y_batch_all = self.pack_to_batch(X_train, Y_train, hyper_params['batch_size'], n_samples)
        
        train_loss_arr = []
        val_loss_arr = []
        
        val_acc_arr = []

        for i in range(hyper_params['epoch']):
            loss = 0
            print("Epoch: ", i)
            with tqdm(total=len(X_batch_all)) as pbar:
                for idx, (X_batch, Y_batch) in enumerate(zip(X_batch_all, Y_batch_all)):
                    # 單個 batch 訓練過程
                    # 1. 前向傳播
                    # 2. 反向傳播
                    # 3. 更新權重   
                    self.forward(X_batch)
                    self.backward(X_batch, Y_batch)
                    self.update_params({'lr': hyper_params['lr'], 'alpha': hyper_params['alpha']})
                    loss += loss_func.cal_loss(self.get_pred(X_batch, with_onehot=True), Y_batch)
                    pbar.update(1)
            print("Epoch: ", i)
            print('Loss:', round(loss, 2) / hyper_params['batch_size'])

            predictions = self.get_pred(X_val)
            print('Val Acc:', round(self.calculate_acc(predictions, Y_val), 2))
            
            train_loss_arr.append(loss / n_samples)

            # 取 output layer 經過 activation function 的結果為 prediction
            val_loss_arr.append(loss_func.cal_loss(self.get_pred(X_val, with_onehot=True), Y_val) / len(X_val))
            val_acc_arr.append(self.calculate_acc(predictions, Y_val))

        if show_plot:
            self.plot_loss_acc(train_loss_arr, val_loss_arr, val_acc_arr)

        return train_loss_arr, val_loss_arr, val_acc_arr
    
    def train_with_dataset(self, train_dataset, loss_func, hyper_params: dict, show_plot=False):
   
        
        train_loss_arr = []
        val_loss_arr = []
        
        val_acc_arr = []
        # 定義轉換
        

        # 載入訓練和測試資料集
        


        # split train and val
        split_ratio = 0.8
        split_idx = int(len(train_dataset) * split_ratio)
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [split_idx, len(train_dataset) - split_idx])

        # 建立資料加載器
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=hyper_params['batch_size'], shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=hyper_params['batch_size'], shuffle=False)

        train_samples = train_loader.dataset.dataset.data.shape[0]
        val_samples = val_loader.dataset.dataset.data.shape[0]

        with torch.no_grad():
            for i in range(hyper_params['epoch']):
                loss = 0
                val_loss = 0
                val_acc = 0
                print("Epoch: ", i)
                with tqdm(total=len(train_loader), desc='train:') as pbar:
                    for idx, (X_batch, Y_batch) in enumerate(train_loader):
                        # 單個 batch 訓練過程
                        # 1. 前向傳播
                        # 2. 反向傳播
                        # 3. 更新權重   
                        
                        Y_batch = torch.eye(10)[Y_batch]
                        X_batch, Y_batch = X_batch.cuda(), Y_batch.cuda()
                        self.forward(X_batch)
                        self.backward(X_batch, Y_batch)
                        self.update_params({'lr': hyper_params['lr'], 'alpha': hyper_params['alpha']})
                        loss += loss_func.cal_loss(self.get_pred(X_batch, with_onehot=True), Y_batch)
                        pbar.update(1)
                print("Epoch: ", i)
                print('Loss:', torch.round(loss, decimals=2) / hyper_params['batch_size'])

                with tqdm(total=len(val_loader), desc='val:') as pbar:
                    for idx, (X_batch, Y_batch) in enumerate(val_loader):
        
                        Y_batch = torch.eye(10)[Y_batch]
                        X_batch, Y_batch = X_batch.cuda(), Y_batch.cuda()

                        predictions = self.get_pred(X_batch, with_onehot=True)
                        val_loss += loss_func.cal_loss(predictions, Y_batch)
                        val_acc += torch.round(self.calculate_acc(predictions, Y_batch), decimals=2)
                        pbar.update(1)

                train_loss_arr.append(loss / train_samples)
                val_loss_arr.append(val_loss / val_samples)

                val_acc_arr.append(val_acc / len(val_loader))
                print('Val Acc:', val_acc / len(val_loader))

            if show_plot:
                self.plot_loss_acc(train_loss_arr, val_loss_arr, val_acc_arr)

        return train_loss_arr, val_loss_arr, val_acc_arr
    

    def calculate_acc(self, predictions, Y):
        Y = torch.argmax(Y, axis=1)
        predictions = torch.argmax(predictions, axis=1)
        return torch.sum(predictions == Y) / len(Y)
    
    def pack_to_batch(self, X, Y, bs, n_samples):

        # 將全部的資料打包成 batch，每個 batch 的大小為 bs
        # 若 n_samples 不能被 bs 整除，則將 X_train, Y_all 進行 padding
        n_dim = len(X.shape)
        if X.shape[0] % bs != 0:  
            X = np.pad(X, [[0, bs - (n_samples % bs)] if i == 0 else [0, 0] for i in range(n_dim)], 'constant', constant_values=(0))
            Y = np.pad(Y, [[0, bs - (n_samples % bs)], [0, 0]], 'constant', constant_values=(0))

        if n_dim == 2:
            X_batch_all = X.reshape(-1, bs, X.shape[1])
            Y_batch_all = Y.reshape(-1, bs, Y.shape[1])
        elif n_dim == 4:
            X_batch_all = X.reshape(-1, bs, X.shape[1], X.shape[2], X.shape[3])
            Y_batch_all = Y.reshape(-1, bs, Y.shape[1])
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