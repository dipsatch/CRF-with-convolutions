import numpy as np


class Conv:

    def __init__(self, K=None, stride=1, padding=0):
        self.kernel_size = K.shape
        self.weights = K.reshape(K.shape[0]*K.shape[1], 1)
        self.biases = np.zeros(1)
        self.stride = stride
        self.padding = padding

    def forward(self,x):
        mb, ch, n, p = x.shape
        y = np.matmul(self.arr2vec(x,self.kernel_size,self.stride,self.padding), self.weights) + self.biases
        print(y.shape)
        y = np.transpose(y,(0,2,1))
        # print(y)
        n1 = (n-self.kernel_size[0]+ 2 * self.padding) //self.stride + 1
        p1 = (p-self.kernel_size[1]+2 * self.padding )//self.stride + 1
        return y.reshape(mb,self.biases.shape[0],n1,p1)

    def arr2vec(self, x, kernel_size, stride=1, padding=0):
        k1, k2 = kernel_size
        mb, ch, n1, n2 = x.shape
        y = np.zeros((mb, ch, n1 + 2 * padding, n2 + 2 * padding))
        y[:, :, padding:n1 + padding, padding:n2 + padding] = x
        start_idx = np.array([j + (n2 + 2 * padding) * i for i in range(0, n1 - k1 + 1 + 2 * padding, stride) for j in
                              range(0, n2 - k2 + 1 + 2 * padding, stride)])
        grid = np.array(
            [j + (n2 + 2 * padding) * i + (n1 + 2 * padding) * (n2 + 2 * padding) * k for k in range(0, ch) for i in
             range(k1) for j in range(k2)])
        to_take = start_idx[:, None] + grid[None, :]
        batch = np.array(range(0, mb)) * ch * (n1 + 2 * padding) * (n2 + 2 * padding)
        return y.take(batch[:, None, None] + to_take[None, :, :])