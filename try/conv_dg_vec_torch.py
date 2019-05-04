import torch

class Conv(torch.nn.Module):

    def __init__(self, K=None, stride=1, padding=0, device=torch.device('cpu')):
        super(Conv, self).__init__()
        self.kernel_size = K.shape
        self.weights = K.view(K.shape[0]*K.shape[1], 1)
        self.biases = torch.zeros(1)
        self.stride = stride
        self.padding = padding
        self.device = device

    def forward(self,x):
        mb, ch, n, p = x.shape
        y = self.arr2vec(x, self.kernel_size, self.stride, self.padding).matmul(self.weights) + self.biases
        y = y.permute(0,2,1)
        n1 = (n-self.kernel_size[0]+ 2 * self.padding) //self.stride + 1
        p1 = (p-self.kernel_size[1]+2 * self.padding )//self.stride + 1
        return y.view(mb,self.biases.shape[0],n1,p1)

    def arr2vec(self, x, kernel_size, stride=1, padding=0):
        k1, k2 = kernel_size
        mb, ch, n1, n2 = x.shape
        y = torch.zeros((mb, ch, n1 + 2 * padding, n2 + 2 * padding)).to(self.device)
        y[:, :, padding:n1 + padding, padding:n2 + padding] = x
        start_idx = torch.tensor([j + (n2 + 2 * padding) * i for i in range(0, n1 - k1 + 1 + 2 * padding, stride) for j in range(0, n2 - k2 + 1 + 2 * padding, stride)]).to(self.device)
        grid = torch.tensor([j + (n2 + 2 * padding) * i + (n1 + 2 * padding) * (n2 + 2 * padding) * k for k in range(0, ch) for i in range(k1) for j in range(k2)]).to(self.device)
        to_take = start_idx[:, None] + grid[None, :]
        batch = torch.tensor(range(0, mb)) * ch * (n1 + 2 * padding) * (n2 + 2 * padding).to(self.device)
        return y.take(batch[:, None, None] + to_take[None, :, :])