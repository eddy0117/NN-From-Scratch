import numpy as np
import matplotlib.pyplot as plt

class CrossEntropyLoss:
    def cal_loss(y, label):
        return -np.sum(label * np.log(y))
    
class SquareLoss:
    def cal_loss(y, label):
        return np.sum((1 / 2) * ((y - label) ** 2))

class MLP:
    def __init__(self, params_set_list: list):
    
        self.params_set_list = params_set_list
        self.params = self.weight_init(params_set_list)

        # momentum 初始化動量為 0
        self.velocity = {
            'w': [np.zeros_like(w) for w in self.params['w']],
            'b': [np.zeros_like(b) for b in self.params['b']]
        }

    def weight_init(self, params_set_list: list) -> dict:
        params = {'w' : [], 'b': [], 'act_func': []}
        for in_features, out_features, act_func in params_set_list:
            # 使用 He initalization 初始化 weight
            # source: https://www.numpyninja.com/post/weight-initialization-techniques

            # w 是高斯分佈的隨機數，所以使用 He init 時分子為 2
            w = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features)
            b = np.zeros((1, out_features))
            params['w'].append(w)
            params['b'].append(b)
            params['act_func'].append(act_func)
        return params
    
    def update_params(self, params: dict, params_delta: dict, lr, alpha) -> dict:


        for idx, (w, b, dW, db) in enumerate(zip(params['w'], params['b'], 
                                                 params_delta['W'][::-1], 
                                                 params_delta['b'][::-1])):
            # 從 input layer 開始更新 params，所以將 dW, db list 反轉
            # 使用 momentum 方法
            # 將上一次的動量用來更新當前動量
            self.velocity['w'][idx] = alpha * self.velocity['w'][idx] - lr * dW
            self.velocity['b'][idx] = alpha * self.velocity['b'][idx] - lr * db
            
            # 更新參數
            params['w'][idx] = w + self.velocity['w'][idx]
            params['b'][idx] = b + self.velocity['b'][idx]


        return params

    def forward(self, params: dict, X) -> dict:
        forward_saved = {'I': [], 'Y': []}
        # X shape: (bs, n_features)

        for idx, (w, b, act_func) in enumerate(zip(params['w'], params['b'], params['act_func'])):
            if idx == 0:
                I = np.matmul(X, w) + b
            else:
                I = np.matmul(Y, w) + b
            Y = act_func.forward(I)
            forward_saved['I'].append(I)
            forward_saved['Y'].append(Y)

        return forward_saved

    def backward(self, params: dict, forward_val: dict, input_feat, label) -> dict:

        params_delta = {'W': [], 'b': []}

        # 設定標準化因子 (batch size的倒數) 將梯度標準化，避免 batch size 過大造成梯度累積過大
        bs = input_feat.shape[0]
        norm_factor = 1 / bs
        num_layers = len(params['w'])
        
        for idx, (Y, I, w, act_func) in enumerate(zip(forward_val['Y'][::-1], 
                                                      forward_val['I'][::-1],
                                                      params['w'][::-1], params['act_func'][::-1])):
            # 由 output layer 開始計算 delta params，所以將 forward_val & params 反轉

            if idx == 0:
                # 當目前為 output layer 時
                # delta = dL/dY * dY/dI 在 Cross Entropy Loss 與 Softmax 的的情況下
                # 可化簡為 Y - label
                # source: https://levelup.gitconnected.com/killer-combo-softmax-and-cross-entropy-5907442f60ba
                delta = (Y - label) * act_func.backward(I)
            else:
                # delta[n+1] -> (bs, n_layer[n+1])
                # next_layer_w.T -> (n_layer[n+1], n_layer[n])
                delta = np.matmul(delta, next_layer_w.T) * act_func.backward(I)
                # delta[n] -> (bs, n_layer[n])

            
            if idx == num_layers - 1:
                # 當目前為 input layer 時，Y 為輸入特徵
                prev_layer_Y = input_feat
            else:
                prev_layer_Y = forward_val['Y'][::-1][idx+1]
         
            # prev_layer_Y.T -> (n_layer[n-1], bs), delta -> (bs, n_layer[n])
            dW = norm_factor * np.matmul(prev_layer_Y.T, delta)    

            # d_bias 實際上是 delta 對一個全部為 1 的矩陣做矩陣乘法，所以直接化簡為 summation
            db = norm_factor * np.sum(delta, axis=0, keepdims=True)
    
            params_delta['W'].append(dW)
            params_delta['b'].append(db)
            next_layer_w = w

        return params_delta

    def get_pred(self, X, with_onehot=False):
        pred = self.forward(self.params, X)
        if with_onehot:
            return pred['Y'][-1]
        return np.argmax(pred['Y'][-1], axis=1)

    def calculate_acc(self, predictions, Y):
        Y = np.argmax(Y, axis=1)
        return np.sum(predictions == Y) / len(Y)
    
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
                forward_saved = self.forward(self.params, X_batch)
                params_delta = self.backward(self.params, forward_saved, X_batch, Y_batch)
                self.params = self.update_params(self.params, params_delta, hyper_params['lr'], hyper_params['alpha'])
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

    def kfold(self, X, Y, k, loss_func, hyper_params: dict):
        n_samples = X.shape[0]

        # 每個 fold 的大小
        fold_size = n_samples // k

        # 打亂數據
        idx = np.random.permutation(n_samples)
        X = X[idx]
        Y = Y[idx]

        all_f_train_loss = []
        all_f_val_loss = []
        all_f_val_acc = []

        for i in range(k):
            print(f'================= Fold {i+1} =================')
            X_val = X[i * fold_size: (i+1) * fold_size]
            Y_val = Y[i * fold_size: (i+1) * fold_size]

            X_train = np.concatenate([X[:i * fold_size], X[(i+1) * fold_size:]])
            Y_train = np.concatenate([Y[:i * fold_size], Y[(i+1) * fold_size:]])
            
            # 每個 fold 要將 params 重新 init
            self.params = self.weight_init(self.params_set_list)
            train_loss_arr, val_loss_arr, val_acc_arr = self.train(X_train, Y_train, X_val, Y_val, loss_func, hyper_params)

            all_f_val_acc.append(val_acc_arr)

            self.plot_loss_acc(train_loss_arr, val_loss_arr, val_acc_arr)
           

        plt.figure(figsize=(7, 4))
        plt.grid()
        plt.title('All folds Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Val Acc')
        for i in range(k):
            plt.plot(all_f_val_acc[i])
        plt.legend([f'Fold {i+1}' for i in range(k)], loc='lower right')
        plt.show()

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

def visualize_features(data, f1, f2, labels):
    plt.set_cmap('viridis')
    plt.figure(figsize=(5, 4))
    # plt.grid()
    plt.xlabel(f'feature {f1}')
    plt.ylabel(f'feature {f2}')
    plt.scatter(data[:, f1], data[:, f2], c=labels, s=5)
    plt.show()