import numpy as np
from ..core import BaseModule

class Conv2d(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.w, self.b, self.velocity = self.weight_init()
        self.params_delta = {'dW': None, 'db': None}
       
        
    def weight_init(self) -> dict:
        params = {}
        velocity = {}
        # 使用 xavier initalization 初始化 weight
        # source: https://www.numpyninja.com/post/weight-initialization-techniques

        # w 是高斯分佈的隨機數，所以使用 xavier init 時分子為 2
        kh, kw = self.kernel_size
        # w = np.random.randn(self.out_channels, self.in_channels, kh, kw) * np.sqrt(2 / (self.in_channels * kh * kw))
        w = (np.random.randn(self.in_channels * kh * kw, self.out_channels) * np.sqrt(2 / (self.in_channels * kh * kw))).\
            reshape(self.out_channels, self.in_channels, kh, kw)
        b = np.zeros((1, self.out_channels))
  
        velocity['w'] = np.zeros_like(w)
        velocity['b'] = np.zeros_like(b)
        return w, b, velocity
    
    def update_params(self, opt_params: dict):
        self.velocity['w'] = opt_params['alpha'] * self.velocity['w'] - opt_params['lr'] * self.params_delta['dW']
        self.velocity['b'] = opt_params['alpha'] * self.velocity['b'] - opt_params['lr'] * self.params_delta['db']
        self.w += self.velocity['w']
        self.b += self.velocity['b']
    
    def forward(self, x): 
        self.in_feat = x
        I = self.convolution(x, self.w, self.b)
        return I
    
    def backward(self, delta):
        #  back_prop_params: {'delta_next': delta_next, 
        #                     'w_next': w_next, 
        #                     'prev_Y': prev_Y,}
        
        #  如果 next layer 是 flatten
        #  dLdZ -> (N, out_c*out_h*out_w)
        dLdZ = self.cal_dLdZ(delta, self.w)
        dW, db = self.cal_dLdW(self.in_feat, delta)
        self.params_delta['dW'] = dW
        self.params_delta['db'] = db
        
        
        
        return dLdZ
    
    def cal_dLdZ(self, delta, filter):
        # delta: (N, out_c, out_h, out_w)
        # filter: (out_c, in_c, kh, kw)

        # assume kh = kw
        dilated_padding = self.kernel_size[0] - 1
        dilated_stride = self.stride

        # rotate filter 180 deg
        filter = np.rot90(filter, 2, axes=(2, 3))

        # do dilation and padding
        dilated_delta = self.dilate(delta, dilated_stride, dilated_padding)

        dLdZ = self.convolution(dilated_delta, filter)

        return dLdZ

    def cal_dLdW(self, x, delta):
        # delta: (N, out_c, out_h, out_w)
        # x: (N, C, H, W)
        diliated_stride = self.stride
        dilated_delta = self.dilate(delta, diliated_stride, 0)
        dLdW = self.convolution(x, dilated_delta)
        # dLdW -> (out_c, in_c, kh, kw)
        dLdb = np.sum(delta, axis=(1, 2, 3))
        # dLdb -> (out_c, 1)
        return dLdW, dLdb

    def dilate(self, x: np.ndarray, stride, padding=None):
        # dilate first, then padding
        # x -> (N, C, H, W)
        dilated_h = x.shape[2] + (x.shape[2] - 1) * (stride - 1)
        dilated_w = x.shape[3] + (x.shape[3] - 1) * (stride - 1)
        dilated_feat = np.zeros((x.shape[0], x.shape[1], dilated_h, dilated_w), dtype=x.dtype)

        if stride:
            dilated_feat[:, :, ::stride, ::stride] = x
        else:
            dilated_feat = x
        if padding:
            dilated_feat = self.zero_padding(dilated_feat, padding)

        return dilated_feat
    
    def zero_padding(self, x: np.ndarray, padding):
        # x -> (N, C, H, W)
        return np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant', constant_values=0)
    
    def im2col(self, input_feat: np.ndarray, N, kh, kw, out_h, out_w, stride):
        im2col_feat = []
        for n in range(N):
            for ih in range(out_h):
                for iw in range(out_w):
                    im2col_feat.append(input_feat[n, :, stride * ih:stride * ih + kh, stride * iw:stride * iw + kw])
                    # each element -> (C, kh, kw)
        # input_feat -> (N*out_h*out_w, C, kh, kw)

        return np.array(im2col_feat).reshape(N * out_h * out_w, -1)

    def convolution(self, input_feat: np.ndarray, filter: np.ndarray, bias=None):
        '''
        input_feat: (N, C, H, W)
        filter: (out_c, C*kh*kw)
        bias: (out_C, 1)
        '''
        N, C, H, W = input_feat.shape
        kh, kw = self.kernel_size
        out_h = int((H - kh + 2 * self.padding) // self.stride) + 1
        out_w = int((W - kw + 2 * self.padding) // self.stride) + 1
        out_c = self.out_channels
        
        if self.padding:
            input_feat = np.pad(input_feat, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=0)

        im2col_feat = self.im2col(input_feat, N, kh, kw, out_h, out_w, self.stride)
        # im2col -> (N*out_h*out_w, C*kh*kw)

        # self.w = self.w.reshape(out_c, -1)
        # filter -> (out_c, C*kh*kw)

        # x @ w
        # x-> (N*out_h*out_w, C*kh*kw)
        # w -> (C*kh*kw, out_c)
        # print(out_h, out_w)
        # print(im2col_feat.shape, filter.shape)
        filter = filter.reshape(-1, out_c)
        if isinstance(bias, np.ndarray):
            out_feat = (im2col_feat @ filter + bias).T
        else:
            out_feat = (im2col_feat @ filter).T
        # out_feat -> (out_c, N*out_h*out_w)
        
        # 將 w 重新 reshape 成 (out_c, in_c, kh, kw)
        # self.w = self.w.reshape(out_c, C, kh, kw)

        # 直接將 (out_c, N*out_h*out_w) reshape 成 (N, out_c, out_h, out_w) 會產生順序錯亂
        # 所以先將 (out_c, N*out_h*out_w) 拆成 (out_c, N, out_h, out_w) 後再 permute
        # out_feat -> (N, out_c, out_h, out_w)
        return out_feat.reshape(out_c, N, out_h, out_w).transpose(1, 0, 2, 3)
        # return out_feat.T.reshape(N, out_h, out_w, out_c).transpose(0, 3, 1, 2)