import torch
from conv_dg import Conv
import numpy as np
import sys


class CRF(torch.nn.Module):
    def __init__(self, input_dim, conv_layers, num_labels, batch_size):
        """
        Linear chain CRF as in Assignment 2
        """
        super(CRF, self).__init__()

        self.input_dim = input_dim
        self.crf_input_dim = None  # Calculated inside init_params() based on Conv layer I/O dims
        self.conv_layers = conv_layers
        self.num_labels = num_labels
        self.batch_size = batch_size
		#Uncomment below and remove self.use_cuda=False once all tensors are set to run on cuda
        #self.use_cuda = torch.cuda.is_available()
        self.use_cuda=False
        self.conv_objects = torch.nn.ModuleList()  # Stores multiple Conv objects
        self.Ks = []   # Stores multiple Conv filters
        self.W = None  # CRF parameter (node potential)
        self.T = None  # CRF parameter (transition weights between node pairs)

        ### Use GPU if available
        if self.use_cuda:
            [m.cuda() for m in self.modules()]

    def init_params(self, Ks=None, W=None, T=None):  # Define trainable parameters here, should be nn.Parameter or with requires_grad=True
        if Ks is None:    # Initialize default Conv filters if nothing is passed via arguments
            for conv_layer in self.conv_layers:
                self.Ks.append(torch.nn.Parameter(torch.ones(conv_layer['filter_shape'], dtype=torch.float)))
        else:
            self.Ks = Ks  # Initialize Conv filters with trained filters if passed via arguments

        # Define input & output dims of each Conv layer & store inside self.conv_layers. Formula: OutputDim = 1 + (InputDim − F + 2P)/S
        last_output_dim = self.input_dim.copy()
        for i, conv_layer in enumerate(self.conv_layers):
            self.conv_objects.append(Conv(K=self.Ks[i], padding=conv_layer['padding'], stride=conv_layer['stride']))  # Instantiate a new Conv object & append in ModuleList to register
            conv_layer['input_dim'] = last_output_dim.copy()
            output_dim_height = int(1 + (conv_layer['input_dim']['height'] - conv_layer['filter_shape'][-2] + 2*conv_layer['padding'])/conv_layer['stride'])
            output_dim_width  = int(1 + (conv_layer['input_dim']['width']  - conv_layer['filter_shape'][-1] + 2*conv_layer['padding'])/conv_layer['stride'])
            output_dim_flattened = output_dim_height * output_dim_width
            conv_layer['output_dim'] = {'flattened': output_dim_flattened, 'height': output_dim_height, 'width': output_dim_width}
            last_output_dim = conv_layer['output_dim'].copy()

        # Define W with the flattened output dim of the last Conv filter that is getting stored in self.crf_input_dim
        self.crf_input_dim = last_output_dim['flattened']
        self.W = torch.nn.Parameter(torch.zeros((self.num_labels, self.crf_input_dim), dtype=torch.float)) if W is None else W
        self.T = torch.nn.Parameter(torch.zeros((self.num_labels, self.num_labels), dtype=torch.float)) if T is None else T

    def get_conv_feats(self, X):
        """
        Generate convolution features for a given word
        """
        convfeatures = X
        for i, conv_layer in enumerate(self.conv_layers):
            convfeatures = self.conv_objects[i](convfeatures.view(convfeatures.shape[:-1] + (conv_layer['input_dim']['height'], conv_layer['input_dim']['width'])))  # Extend the last dim of X, e.g. 128 to (16,8)
            convfeatures = convfeatures.view(convfeatures.shape[:-2] + (convfeatures.shape[-1] * convfeatures.shape[-2],))  # Flatten last 2 dims, e.g. (16,8) to 128

        return convfeatures

    def predict(self, X):  # returns predicted sequence
        features = self.get_conv_feats(X)
        W = self.W
        T = self.T
        prediction = self.crf_decode(W, T, features)
        return prediction

    def forward(self, X, labels):
        """
        Implement the objective of CRF here.
        The input (features) to the CRF module should be convolution features.
        """
        features = self.get_conv_feats(X)
        W = self.W
        T = self.T
        log_prob = CRFautograd.apply(W, T, features, labels)
        return log_prob

    def loss(self, log_prob, C):
        """
        Compute the negative conditional log-likelihood of a labelling given a sequence.
        """
        W = self.W
        T = self.T
        average_log_loss = -C * log_prob
        W_norm = torch.sum(torch.tensor([(torch.norm(Wy.float())) ** 2 for Wy in W])) / 2
        T_norm = torch.sum(torch.tensor([torch.sum(torch.tensor([Tij ** 2 for Tij in row])) for row in T])) / 2
        loss = average_log_loss + W_norm + T_norm
        return loss

    def crf_decode(self, W, T, X_train):
        y_pred = []
        n = self.num_labels
        m = self.crf_input_dim

        def maxSumBottomUp(X, W, T):
            l = torch.zeros((m, n))
            opts = torch.zeros((m, n))
            yStar = torch.zeros(m, dtype=torch.int8)
            for i in range(1, m):
                for a in range(n):
                    tmp = torch.zeros(n)
                    for b in range(n):
                        tmp[b] = torch.dot(X[i - 1], W[b]) + T[a, b] + l[i - 1, b]
                    l[i, a] = max(tmp)
            for b in range(n):
                opts[m - 1, b] = torch.dot(X[m - 1], W[b]) + l[m - 1, b]
            MAP = max(opts[m - 1])
            yStar[m - 1] = opts[m - 1].max(0)[1]
            for i in range(m - 1, 0, -1):
                for b in range(n):
                    opts[i - 1, b] = torch.dot(X[i - 1], W[b]) + T[yStar[i], b] + l[i - 1, b]
                yStar[i - 1] = opts[i - 1].max(0)[1]
            return MAP, yStar

        for X in X_train:
            target = torch.zeros((m, 26), dtype=torch.int8)
            MAP, yStar = maxSumBottomUp(X, W, T)
            for i, j in enumerate(yStar):
                target[i][j - 1] = 1
            y_pred.append(target)
        return torch.stack(y_pred)


class CRFautograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W, T, X_train, y_train):
        ctx.save_for_backward(W, T, X_train, y_train)
        return torch.tensor([CRFautograd.compute_log_probability(X, y, W, T) for X, y in zip(X_train, y_train)]).sum() / len(X_train)

    @staticmethod
    def backward(ctx, grad_out):
        # print('Bwd called')
        W, T, X, y = ctx.saved_tensors
        grad_Ws, grad_Ts = CRFautograd.crf_obj_prime(W, T, X, y)
        grad_Ws = grad_out * grad_Ws
        grad_Ts = grad_out * grad_Ts
        return grad_Ws, grad_Ts, torch.ones(X.shape), None  # TODO: Replace 3rd with d(loss)/d(X), where X is CRF input

    @staticmethod
    def forward_messages(X, W, T):
        n = len(T)  # 26 letters
        m = len(X)  # word length
        f_msgs = torch.zeros((m, n), dtype=torch.float)
        # Implement the code from both of the comments in section 5.1 of the appendix
        for i in range(n):
            f_msgs[0, i] = torch.exp(torch.dot(X[0], W[i]).float())
        for i in range(1, m):
            for a in range(n):
                product = torch.exp(torch.dot(X[i], W[a]).float())
                tmp = torch.zeros(n)
                for b in range(n):
                    tmp[b] = torch.exp(T[b, a]) * f_msgs[i - 1, b]
                f_msgs[i, a] = product * torch.sum(tmp)
        return f_msgs

    @staticmethod
    def backward_messages(X, W, T):
        n = len(T)  # 26 letters
        m = len(X)  # word length
        b_msgs = torch.zeros((m, n), dtype=torch.float)
        for i in range(n):
            b_msgs[-1, i] = torch.exp(torch.dot(X[-1], W[i]).float())
        for i in range(m - 2, -1, -1):
            for a in range(n):
                product = torch.exp(torch.dot(X[i], W[a]).float())
                tmp = torch.zeros(n)
                for b in range(n):
                    tmp[b] = torch.exp(T[b, a]) * b_msgs[i + 1, b]
                b_msgs[i, a] = product * torch.sum(tmp)
        return b_msgs

    @staticmethod
    def compute_grad_W(X, y, W, T):
        n = len(T)
        m = len(X)
        grad_W = torch.zeros(W.shape, dtype=torch.float)
        f_msgs = CRFautograd.forward_messages(X, W, T)
        b_msgs = CRFautograd.backward_messages(X, W, T)
        Z = torch.sum(f_msgs[-1])

        for i in range(n):
            tmp = X[0] * b_msgs[0, i] + X[-1] * f_msgs[-1, i]
            for j in range(1, m - 1):
                tmp += X[j] * (f_msgs[j, i] * b_msgs[j, i] / torch.exp(torch.dot(W[i], X[j])))
            grad_W[i] = sum([X[s] for s in range(m) if y[s] == i]) - tmp / Z

        return grad_W

    @staticmethod
    def compute_grad_T(X, y, W, T):
        n = len(T)
        m = len(X)
        grad_T = torch.zeros(T.shape, dtype=torch.float)
        f_msgs = CRFautograd.forward_messages(X, W, T)
        b_msgs = CRFautograd.backward_messages(X, W, T)
        Z = torch.sum(f_msgs[-1])

        for i in range(1, m):
            for j in range(n):
                for k in range(n):
                    grad_T[j, k] -= f_msgs[i - 1, j] * b_msgs[i, k] * torch.exp(T[j, k])
        grad_T = grad_T / Z
        for i in range(1, m):
            grad_T[y[i - 1], y[i]] += 1

        return grad_T

    @staticmethod
    def crf_obj_prime(W, T, X_data, y_data):
        n = len(T)
        dim = W.shape[1]
        grad_Ws = torch.zeros((n, dim), dtype=torch.float)
        grad_Ts = torch.zeros((n, n), dtype=torch.float)
        for X, y in zip(X_data, y_data):
            y = torch.nonzero(y)[:,1].numpy()
            X = X[:len(y)]
            grad_Ws += CRFautograd.compute_grad_W(X, y, W, T)
            grad_Ts += CRFautograd.compute_grad_T(X, y, W, T)

        grad_Ws = grad_Ws / float(len(X_data))
        grad_Ts = grad_Ts / float(len(X_data))

        return grad_Ws, grad_Ts

    @staticmethod
    def compute_log_probability(X, y, W, T):
        y = torch.nonzero(y)[:,1].numpy()
        X = X[:len(y)]
        Z = torch.sum(CRFautograd.forward_messages(X, W, T)[-1])
        unnormalized = torch.sum(X.mm(W[y].t()))
        unnormalized += torch.sum(torch.tensor([T[y[i], y[i + 1]] for i in range(len(X) - 1)]))
        return torch.log(torch.exp(unnormalized) / Z)
