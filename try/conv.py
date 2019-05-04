import torch
import torch.nn as nn

class Conv(nn.Module):
	"""
	Convolution layer.
	"""
	def __init__(self,X,K,b=0,padding=0,stride=1):
		super(Conv, self).__init__()
		self.init_params(X,K,b,padding,stride)
	
	def init_params(self,X,K,b=0,padding=0,stride=1):
		self.X=X
		self.K=K
		self.padding=padding
		self.stride=stride
		if b==0:
			self.b=torch.zeros([1,1,1,1], dtype=torch.float, requires_grad=True)
		else:
			self.b=b

	def forward(self,X,K):
		bx,cx,hx,wx=X.size()
		bk,ck,hk,wk=K.size()
		h_out= int((hx-hk + 2*self.padding)/self.stride + 1)
		w_out= int( (wx - wk + 2*self.padding)/self.stride + 1)
		p2d=(int(self.padding),int(self.padding),int(self.padding),int(self.padding))
		X=torch.nn.functional.pad(X, p2d, 'constant', 0)
		Z=torch.zeros([bx,cx,h_out,w_out], dtype=torch.float)
		for b in range(0,bx):
			for c in range(0,cx):
				for h in range(0,h_out,self.stride):
					for w in range(0,w_out,self.stride):	
						Z[b,c,h,w]=torch.sum(X[b,c,h:h+hk,w:w+wk]*K + self.b)
		cache=(X,K,self.b,self.padding,self.stride)			
		
		return Z,cache

		
		




