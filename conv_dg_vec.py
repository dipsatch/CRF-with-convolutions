import torch

class Conv(torch.nn.Module):

    def __init__(self, K, padding=0, stride=1, device=torch.device('cpu')):
        super(Conv, self).__init__()
        self.kernel_size = K.shape
        self.K = K
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.stride = stride
        self.padding = padding
        self.device = device

    def forward(self, X):
        mb, ch, n, p = X.shape
        y = self.arr2vec(X, self.kernel_size, self.stride, self.padding).matmul(self.K.view(self.K.shape[0]*self.K.shape[1], 1)) + self.bias
        y = y.permute(0,2,1)
        n1 = (n - self.kernel_size[0]+ 2 * self.padding) // self.stride + 1
        p1 = (p - self.kernel_size[1]+2 * self.padding ) // self.stride + 1
        return y.view(mb, self.bias.shape[0], n1, p1)

    def arr2vec(self, x, kernel_size, stride=1, padding=0):
        k1, k2 = kernel_size
        mb, ch, n1, n2 = x.shape
        y = torch.zeros((mb, ch, n1 + 2 * padding, n2 + 2 * padding), device=self.device)
        y[:, :, padding:n1 + padding, padding:n2 + padding] = x
        start_idx = torch.tensor([j + (n2 + 2 * padding) * i for i in range(0, n1 - k1 + 1 + 2 * padding, stride) for j in range(0, n2 - k2 + 1 + 2 * padding, stride)], device=self.device)
        grid = torch.tensor([j + (n2 + 2 * padding) * i + (n1 + 2 * padding) * (n2 + 2 * padding) * k for k in range(0, ch) for i in range(k1) for j in range(k2)], device=self.device)
        to_take = start_idx[:, None] + grid[None, :]
        batch = torch.tensor(range(0, mb), device=self.device) * ch * (n1 + 2 * padding) * (n2 + 2 * padding)
        return y.take(batch[:, None, None] + to_take[None, :, :])