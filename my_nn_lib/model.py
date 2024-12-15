import numpy as np
from tqdm import tqdm
from .my_mlp import MLP



class MyModel(MLP):
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
        return np.argmax(pred, axis=1)

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
    
    