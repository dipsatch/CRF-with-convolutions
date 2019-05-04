import torch

class Conv(torch.nn.Module):
    """
    Convolution layer.
    """
    def __init__(self, K, padding=0, stride=1, b=None, device=torch.device('cpu')):
        super(Conv, self).__init__()
        self.padding = padding
        self.stride = stride
        self.device = device
        self.b = torch.nn.Parameter(torch.zeros((1,1,1,1))) if b is None else b
        self.K = K  # torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(K)) if isinstance(K, tuple) else K)

    def forward(self, X):
        bx, cx, hx, wx = X.size()
        bk, ck, hk, wk = self.K.size()
        h_out = int((hx-hk + 2*self.padding)/self.stride + 1)
        w_out = int((wx - wk + 2*self.padding)/self.stride + 1)
        p2d = (int(self.padding), int(self.padding), int(self.padding), int(self.padding))
        X = torch.nn.functional.pad(X, p2d, 'constant', 0)
        Z = torch.zeros((bx, cx, h_out, w_out)).to(self.device)
        for b in range(0, bx):
            for c in range(0, cx):
                for h in range(0, h_out, self.stride):
                    for w in range(0, w_out, self.stride):
                        Z[b, c, h, w] = torch.sum(X[b, c, h:h+hk, w:w+wk] * self.K + self.b)
        return Z

