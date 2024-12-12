import os
import cv2
import numpy as np
from my_mlp import MLP, CrossEntropyLoss
from my_nn_lib import ReLU, Softmax, LeckyReLU, Linear, BaseModule, Conv2d, Flatten

# import mnist



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
            for idx, (X_batch, Y_batch) in enumerate(zip(X_batch_all, Y_batch_all)):
                print("Batch: ", idx)
                # 單個 batch 訓練過程
                # 1. 前向傳播
                # 2. 反向傳播
                # 3. 更新權重   
                self.forward(X_batch)
                self.backward(X_batch, Y_batch)
                self.update_params({'lr': hyper_params['lr'], 'alpha': hyper_params['alpha']})
                loss += loss_func.cal_loss(self.get_pred(X_batch, with_onehot=True), Y_batch)
              
            # print("Epoch: ", i)
            # print('Loss:', round(loss, 2))

            predictions = self.get_pred(X_val)
            # print('Val Acc:', round(get_accuracy(predictions, Y_val), 2))
            
            train_loss_arr.append(loss / n_samples)

            # 取 output layer 經過 activation function 的結果為 prediction
            val_loss_arr.append(loss_func.cal_loss(self.get_pred(X_val, with_onehot=True), Y_val) / len(X_val))
            val_acc_arr.append(self.calculate_acc(predictions, Y_val))

        if show_plot:
            self.plot_loss_acc(train_loss_arr, val_loss_arr, val_acc_arr)

        return train_loss_arr, val_loss_arr, val_acc_arr
    

if __name__ == "__main__":

    path = 'data/MNIST'

    class_paths = os.listdir(path)

    x_all = []
    y_all = []

    for cls_path in class_paths:
        img_paths = os.listdir(os.path.join(path, cls_path))
        for img_path in img_paths:
            img = cv2.imread(os.path.join(path, cls_path, img_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))
            x_all.append(img)
            y_all.append(int(cls_path))
    x_all = np.array(x_all)
    y_all = np.array(y_all)
    
    x_all = x_all.reshape(-1, 1, 28, 28) / 255

    # one-hot encoding
    y_one_hot = np.zeros((len(y_all), 10))
    y_one_hot[np.arange(len(y_all)), y_all] = 1

    # shuffle data
    idx = np.arange(len(x_all))
    np.random.shuffle(idx)
    x_all = x_all[idx]
    y_one_hot = y_one_hot[idx]

    

    # split train and val
    split_ratio = 0.8
    split_idx = int(len(x_all) * split_ratio)
    x_train = x_all[:split_idx]
    y_train = y_one_hot[:split_idx]
    x_val = x_all[split_idx:]
    y_val = y_one_hot[split_idx:]

    # making some dummy data about 3 images with 3 channels and labels
    # images = np.random.randint(0, 255, size=(4, 3, 28, 28)).astype(np.float32) / 255
    # labels = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])

    model = MyModel([Conv2d(1, 4, (3, 3), 1, 0), 
                    LeckyReLU(),
                    Conv2d(4, 2, (3, 3), 1, 0),
                    LeckyReLU(),
                    Flatten(),
                    Linear(2*24*24, 10),
                    Softmax()])

    hyper_params = {    
        'lr': 0.01,
        'epoch': 2,
        'batch_size': 4096,
        'alpha': 0.9
    }
    model.train(x_train, y_train, x_val, y_val, CrossEntropyLoss, hyper_params, show_plot=True)
    