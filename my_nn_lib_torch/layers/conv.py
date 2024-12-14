import time
import logging
import numpy as np
import torch
import torch.nn.functional as F
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
        w = (torch.randn(self.in_channels * kh * kw, self.out_channels) * np.sqrt(2 / (self.in_channels * kh * kw))).\
            reshape(self.out_channels, self.in_channels, kh, kw).cuda()
        b = torch.zeros((1, self.out_channels)).cuda()
  
        velocity['w'] = torch.zeros_like(w).cuda()
        velocity['b'] = torch.zeros_like(b).cuda()
        return w, b, velocity
    
    def update_params(self, opt_params: dict):
        self.velocity['w'] = opt_params['alpha'] * self.velocity['w'] - opt_params['lr'] * self.params_delta['dW']
        self.velocity['b'] = opt_params['alpha'] * self.velocity['b'] - opt_params['lr'] * self.params_delta['db']
        self.w += self.velocity['w']
        self.b += self.velocity['b']
    
    def forward(self, x): 
        self.in_feat = x
        
        start = time.time()
        I = self.convolution(x, self.w, self.b, stride=self.stride)
        logging.info(f"forward time: {time.time() - start}")
        # print(I.shape)
        return I
    
    def backward(self, delta):
        # delta : (N, out_c, out_h, out_w)
        
        #  如果 next layer 是 flatten
        #  dLdZ -> (N, out_c*out_h*out_w)

        # delta = np.mean(delta, axis=0, keepdims=True)

        dLdZ = self.cal_dLdZ(delta, self.w)
        dW, db = self.cal_dLdW(self.in_feat, delta)
        self.params_delta['dW'] = dW
        self.params_delta['db'] = db
        
        
        
        return dLdZ
    
    def cal_dLdZ(self, delta, filter):
        # delta: (N, convd_c, convd_h, convd_w)
        # filter: (convd_c, ori_c, kh, kw)

        # assume kh = kw
        dilated_padding = self.kernel_size[0] - 1
        dilated_stride = self.stride

        # rotate filter 180 deg
        filter = torch.rot90(filter, 2, dims=(2, 3))
        
        # do dilation and padding
        dilated_delta = self.dilate(delta, dilated_stride, dilated_padding)

        
        dilated_delta = torch.tile(dilated_delta, (1, self.in_channels, 1, 1))
        # dilated_delta -> (N, ori_c*convd_c, convd_h, convd_w)

        dLdZ = self.t_convolution(dilated_delta, filter, stride=1, mode="dLdZ")

        return dLdZ

    def cal_dLdW(self, x, delta):
        # delta: (N, convd_c, convd_h, convd_w)
        # x: (N, ori_c, H, W)
        norm_factor = 1 / x.shape[0]
        diliated_stride = self.stride



        dilated_delta = self.dilate(delta, diliated_stride, 0)
        dilated_delta = dilated_delta.permute(1, 0, 2, 3)
        # dilated_delta -> (convd_c, N, dconvd_h, dconvd_w)
        
        dilated_delta = torch.tile(dilated_delta, (1, self.in_channels, 1, 1))
        # dilated_delta -> (convd_c, ori_c*N, dconvd_h, dconvd_w)

        dLdW = norm_factor * self.t_convolution(x, dilated_delta, stride=1, mode="dLdW")
        # dLdW -> (out_c, in_c, kh, kw)
        
        dLdb = norm_factor * torch.sum(delta, axis=(0, 2, 3))
        # dLdb -> (out_c, 1)
        return dLdW, dLdb

    def dilate(self, x, stride, padding=None):
        # dilate first, then padding
        # x -> (N, C, H, W)
        dilated_h = x.shape[2] + (x.shape[2] - 1) * (stride - 1)
        dilated_w = x.shape[3] + (x.shape[3] - 1) * (stride - 1)
        dilated_feat = torch.zeros((x.shape[0], x.shape[1], dilated_h, dilated_w), dtype=x.dtype).cuda()

        if stride:
            dilated_feat[:, :, ::stride, ::stride] = x
        else:
            dilated_feat = x
        if padding:
            dilated_feat = self.zero_padding(dilated_feat, padding)

        return dilated_feat
    
    def zero_padding(self, x, padding):
        # x -> (N, C, H, W)
        return F.pad(x, (padding, padding, padding, padding), mode='constant', value=0)
    
    def im2col(self, input_feat, N, kh, kw, out_h, out_w, stride):
        
      
        # 當前實作使用循環，效率較低
        # 改進方案：使用 numpy 的向量化操作
        input_feat = input_feat.permute(0, 2, 3, 1)  # (N, H, W, C)
        
        # 創建滑動視窗的索引
        i0 = torch.arange(kh).repeat(kw)
        i1 = torch.arange(out_h).repeat_interleave(out_w) * stride
        i2 = torch.arange(kw).repeat(kh)
        i3 = torch.arange(out_w).repeat_interleave(out_h) * stride
        
        # 使用高級索引一次性獲取所有窗口
        col = input_feat[:, i0[None, :] + i1[:, None],
                        i2[None, :] + i3[:, None], :]
        
        # 重塑為所需形狀
        col = col.reshape(N * out_h * out_w, -1)

        return col
        
            
    def convolution(self, input_feat, filter, bias=None, stride=1, mode="standard"):
        '''
        input_feat: (N, C, H, W)
        filter: (out_c, C*kh*kw)
        bias: (out_C, 1)
        '''
        N, C, H, W = input_feat.shape
    

        kh, kw = filter.shape[2:]
        out_h = int((H - kh + 2 * self.padding) // stride) + 1
        out_w = int((W - kw + 2 * self.padding) // stride) + 1
        out_c = filter.shape[0]
        
        if self.padding:
            input_feat = self.zero_padding(input_feat, self.padding)
        start = time.time()
        im2col_feat = self.im2col(input_feat, N, kh, kw, out_h, out_w, stride)
        logging.info(f"im2col time: {time.time() - start}")
        # if is_dLdW:
        #     im2col_feat = im2col_feat.T

        # im2col -> (N*out_h*out_w, in_c*kh*kw)

        # self.w = self.w.reshape(out_c, -1)
        # filter -> (out_c, in_c*kh*kw)

        # x @ w
        # x-> (N*out_h*out_w, in_c*kh*kw)
        # w -> (in_c*kh*kw, out_c)
        # print(out_h, out_w)
        # print(im2col_feat.shape, filter.shape)

        # filter: (out_c, in_c, kh, kw) -transpose-> (in_c, kh, kw, out_c) -reshape-> (in_c*kh*kw, out_c)
        start = time.time()
        filter_trans = filter.permute(1, 2, 3, 0).reshape(-1, out_c)
        if isinstance(bias, np.ndarray):
            
            out_feat = (im2col_feat @ filter_trans  + bias).T
            
        else:
            out_feat = (im2col_feat @ filter_trans).T
        # out_feat -> (out_c, N*out_h*out_w)
    
    
    
        # 將 w 重新 reshape 成 (out_c, in_c, kh, kw)
        # self.w = self.w.reshape(out_c, C, kh, kw)

        # 直接將 (out_c, N*out_h*out_w) reshape 成 (N, out_c, out_h, out_w) 會產生順序錯亂
        # 所以先將 (out_c, N*out_h*out_w) 拆成 (out_c, N, out_h, out_w) 後再 permute
        # out_feat -> (N, out_c, out_h, out_w)
        out_feat = out_feat.reshape(out_c, N, out_h, out_w).permute(1, 0, 2, 3)
        logging.info(f"conv time: {time.time() - start}")
        return out_feat
  
    def t_im2col(self, input_feat, N, kh, kw, out_h, out_w, stride, mode=None):
        input_feat = input_feat.permute(0, 2, 3, 1)  # (N, H, W, C)
        
        # 創建滑動視窗的索引
        i0 = torch.arange(kh).repeat(kw)
        i1 = torch.arange(out_h).repeat_interleave(out_w) * stride
        i2 = torch.arange(kw).repeat(kh)
        i3 = torch.arange(out_w).repeat_interleave(out_h) * stride
        
        # 使用高級索引一次性獲取所有窗口
        im2col_feat = input_feat[:, i0[None, :] + i1[:, None],
                        i2[None, :] + i3[:, None], :]
        if mode == "dLdW":
            # im2col_feat  : (N*kh*kw, ori_c, dconvd_h, dconvd_w) 
            #             -> (ori_c, N*kh*kw, dconvd_h, dconvd_w) 
            #             -> (ori_c, kh*kw , N*dconvd_h*dconvd_w)
            return im2col_feat.permute(1, 0, 2, 3).reshape(-1, out_h * out_w, N * kh * kw)
        elif mode == "dLdZ":
            # convd_c means the output channel of forward prop phase
            # im2col_feat  : (N*ori_h*ori_w, convd_c*ori_c, kh, kw) 
            #             -> (N*ori_h*ori_w, convd_c, ori_c, kh, kw) 
            #             -> (convd_c, ori_c, N*ori_h*ori_w, kh, kw)
            #             -> (convd_c, ori_c, N*ori_h*ori_w, kh*kw)
            ori_c = self.in_channels
            return im2col_feat.reshape(N * out_h * out_w, -1, ori_c, kh, kw)\
                                        .permute(1, 2, 0, 3, 4)\
                                        .reshape(-1, ori_c, N * out_h * out_w, kh * kw)
        else:
            raise NotImplementedError   
        
    def t_convolution(self, input_feat, filter, stride=1, mode=None):

        N, C, H, W = input_feat.shape
    

        kh, kw = filter.shape[2:]
        ori_h = int((H - kh + 2 * self.padding) // stride) + 1
        ori_w = int((W - kw + 2 * self.padding) // stride) + 1
        convd_c = filter.shape[0]
        
        if self.padding:
            input_feat = self.zero_padding(input_feat, self.padding)
        start = time.time()
        im2col_feat = self.t_im2col(input_feat, N, kh, kw, ori_h, ori_w, stride, mode)
        logging.info(f"t_im2col time:{time.time() - start} ")

        # filter: (out_c, in_c, kh, kw) -transpose-> (in_c, kh, kw, out_c) -reshape-> (in_c*kh*kw, out_c)
        start = time.time()
        if mode == "dLdW":
            # im2col_feat: (ori_c, kh*kw , N*dconvd_h*dconvd_w)
            # filter : (convd_c, ori_c*N, dconvd_h, dconvd_w)
            #       -> (convd_c, ori_c, N*dconvd_h*dconvd_w) 
            #       -> (ori_c, N*dconvd_h*dconvd_w, convd_c)
            filter_trans = filter.reshape(convd_c, -1, N * kh * kw).permute(1, 2, 0)
            out_feat = (im2col_feat @ filter_trans)
            # out_feat : (ori_c, kh*kw, convd_c)
            #         -> (ori_c, kh, kw, convd_c)
            #         -> (convd_c, ori_c, kh, kw)
            out_feat = out_feat.reshape(-1, ori_h, ori_w, convd_c).permute(3, 0, 1, 2)
        elif mode == "dLdZ":
            # im2col_feat: (convd_c, ori_c, N*ori_h*ori_w, kh*kw)
            # filter : (convd_c, ori_c, kh, kw) 
            #       -> (convd_c, ori_c, kh*kw, 1) 

            filter_trans = filter.reshape(convd_c, -1, kh * kw, 1)
            out_feat = (im2col_feat @ filter_trans)
            # out_feat : (convd_c, ori_c, N*ori_h*ori_w, 1)
            #         -> (convd_c, ori_c, N, ori_h, ori_w)
            #         -> (convd_c, N, ori_c, ori_h, ori_w)
            # sum0dim -> (N, ori_c, ori_h, ori_w)

            out_feat = out_feat.reshape(convd_c, -1, N, ori_h, ori_w)\
                           .permute(0, 2, 1, 3, 4)\
                           .sum(axis=0)
        else:
            raise NotImplementedError
        logging.info(f"t_conv time: {time.time() - start}")
        return out_feat