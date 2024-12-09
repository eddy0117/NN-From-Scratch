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
            for X_batch, Y_batch in zip(X_batch_all, Y_batch_all):
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
    # making some dummy data about 3 images with 3 channels and labels
    images = np.random.randint(0, 255, size=(4, 3, 7, 7)).astype(np.float32) / 255
    labels = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])

    model = MyModel([Conv2d(3, 1, (3, 3), 2, 0), 
                    LeckyReLU(),
                    # Conv2d(4, 3, (3, 3), 1, 0),
                    # LeckyReLU(),
                    Flatten(),
                    Linear(1*3*3, 3),
                    Softmax()])

    hyper_params = {    
        'lr': 0.01,
        'epoch': 50,
        'batch_size': 2,
        'alpha': 0.9
    }
    model.train(images, labels, images, labels, CrossEntropyLoss, hyper_params, show_plot=True)
    