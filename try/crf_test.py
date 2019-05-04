import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data_utils
from data_loader import get_dataset
from crf import CRF
import sys

if __name__=='__main__':
	torch.multiprocessing.freeze_support()
	# Tunable hyperparameters
	batch_size = 8  # 256
	num_epochs = 10
	max_iters  = 100
	learning_rate = 0.01
	C = 1000  # C > 0 is a trade-off weight that balances log-likelihood and regularization

	# Model arguments used for parameter initialization
	input_dim = {'flattened': 128, 'height': 16, 'width': 8}
	num_labels = 26
	conv_layers = [  # Define the Convolution layers in order. Tune these too.
    {'filter_shape': (1, 1, 5, 5), 'padding': 0, 'stride': 1},
    {'filter_shape': (1, 1, 3, 3), 'padding': 0, 'stride': 1}
	]

	print_iter = 25  # Prints results every n iterations
	cuda = torch.cuda.is_available()  # Use GPU?

	# Instantiate the CRF model	
	crf = CRF(input_dim, conv_layers, num_labels, batch_size)
	crf.init_params()  # Register submodules & initialize all model parameters (with requires_grad=True)

	# print(list(crf.parameters()))               # Verify if all parameters are listed
	# for name, param in crf.named_parameters():  # Print parameters with names
	#     if param.requires_grad: print(name, param.data)

	opt = torch.optim.LBFGS(crf.parameters(), lr=learning_rate, max_iter=max_iters)

	dataset = get_dataset()
	split = int(0.5 * len(dataset.data)) # train-test split
	train_data, test_data = dataset.data[:split], dataset.data[split:]
	train_target, test_target = dataset.target[:split], dataset.target[split:]

	# Convert dataset into torch tensors
	train = data_utils.TensorDataset(torch.tensor(train_data).float(), torch.tensor(train_target).long())
	test = data_utils.TensorDataset(torch.tensor(test_data).float(), torch.tensor(test_target).long())

	# Define train and test loaders
	train_loader = data_utils.DataLoader(train,  # dataset to load from
                                     batch_size=batch_size,  # examples per batch (default: 1)
                                     shuffle=True,
                                     sampler=None,  # if a sampling method is specified, `shuffle` must be False
                                     num_workers=5,  # subprocesses to use for sampling
                                     pin_memory=False,  # whether to return an item pinned to GPU
                                     )

	test_loader = data_utils.DataLoader(test,  # dataset to load from
                                    batch_size=batch_size,  # examples per batch (default: 1)
                                    shuffle=False,
                                    sampler=None,  # if a sampling method is specified, `shuffle` must be False
                                    num_workers=5,  # subprocesses to use for sampling
                                    pin_memory=False,  # whether to return an item pinned to GPU
                                    )
	print('Loaded dataset... ')

	# Now start training
	for i_batch, sample in enumerate(train_loader):

		train_X = sample[0]
		train_Y = sample[1]
		#Uncomment below once all tensors are set to run on cuda
		#if cuda:
		#	train_X = train_X.cuda()
		#	train_Y = train_Y.cuda()

		def closure():
			opt.zero_grad()

			output = crf(train_X, train_Y)
			print('output---', output)
			loss = crf.loss(output, C)
			print('loss-----', loss)
			print()
			# print(crf.W.grad)
			# print(crf.T.grad)
			# print(crf.Ks[0])
			loss.backward()
			# print(crf.Ks[0].grad)
			return loss

		opt.step(closure)

		sys.exit()
