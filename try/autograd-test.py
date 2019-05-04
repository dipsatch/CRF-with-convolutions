import torch
import torch.autograd as autograd
import torch.nn as nn


class MyFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        b = a.clone()
        c = b*b
        return 2*a

    @staticmethod
    def backward(ctx, grad_out):
        print(grad_out)
        grad_input = grad_out.clone()
        print('Custom backward called!')
        return grad_out * 666


print('***** MyFun(x) ***** \n')
x = autograd.Variable(torch.Tensor([3]), requires_grad=True)
z = MyFun.apply(x)
print(z)
loss = -z - x*x
loss.backward()
print('Gradient', x.grad)  # this is 666, which is correct.

# print('***** MyFun(x+1) ***** \n')
# x = autograd.Variable(torch.Tensor([1]), requires_grad=True)
# z = MyFun.apply(x + 1)
# z.backward()
# print('Gradient', x.grad)  # this should be 666, not 1.