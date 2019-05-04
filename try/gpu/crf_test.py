import torch
import numpy as np
import torch.utils.data as data_utils
from data_loader import get_dataset
from crf import CRF
import sys, time

if __name__=='__main__':
    torch.multiprocessing.freeze_support()
    # Tunable hyperparameters
    batch_size = 256
    num_epochs = 10
    max_iters  = 1000
    learning_rate = 1
    C = 1000  # C > 0 is a trade-off weight that balances log-likelihood and regularization

    # Model arguments used for parameter initialization
    input_dim = {'flattened': 128, 'height': 16, 'width': 8}
    num_labels = 26
    conv_layers = [  # Define the Convolution layers in order. Tune these too.
        {'filter_shape': (5, 5), 'padding': 0, 'stride': 1},
        # {'filter_shape': (3, 3), 'padding': 0, 'stride': 1}
        # {'filter_shape': (1, 1, 5, 5), 'padding': 0, 'stride': 1},
        # {'filter_shape': (1, 1, 3, 3), 'padding': 0, 'stride': 1}
    ]

    print_iter = 25  # Prints results every n iterations
    cuda = torch.cuda.is_available()  # Use GPU?
    device = torch.device('cuda:0' if cuda else 'cpu')

    # Instantiate the CRF model
    crf = CRF(input_dim, conv_layers, num_labels, batch_size, cuda)
    crf.init_params()  # Register submodules & initialize all model parameters (with requires_grad=True)

    # print(list(crf.parameters()))               # Verify if all parameters are listed
    for name, param in crf.named_parameters():  # Print parameters with names
        if param.requires_grad: print(name) #, param.data)

    opt = torch.optim.LBFGS(crf.parameters(), lr=learning_rate, max_iter=5)

    ##################################################
    # Begin training
    ##################################################
    step = 0
    for i in range(num_epochs):
        print("Processing epoch {}".format(i))
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
            # print('Batch:', i_batch)
            train_X = sample[0]
            train_Y = sample[1]

            if cuda:
                train_X = train_X.cuda()
                train_Y = train_Y.cuda()

            def closure():
                opt.zero_grad()
                output = crf(train_X, train_Y)
                print('output---', output)
                tr_loss = crf.loss(output, C)
                print('loss-----', tr_loss)
                # print()
                # print(crf.W.grad)
                # print(crf.T)
                # print(crf.Ks[0])
                tr_loss.backward()
                # print(crf.T.grad)
                # print(crf.Ks[0].grad)
                return tr_loss

            st = time.time()
            tr_loss = opt.step(closure)
            print('Time:', time.time()-st)

            # print to stdout occasionally:
            if step % print_iter == 0:
                random_ixs = np.random.choice(test_data.shape[0], batch_size, replace=False)
                test_X = test_data[random_ixs, :]
                test_Y = test_target[random_ixs, :]

                # Convert to torch
                test_X = torch.from_numpy(test_X).float()
                test_Y = torch.from_numpy(test_Y).long()

                if cuda:
                    test_X = test_X.cuda()
                    test_Y = test_Y.cuda()
                test_loss = crf.loss(crf(train_X, train_Y), C)
                print(step, tr_loss.data, test_loss.data,
                      tr_loss.data / batch_size, test_loss.data / batch_size)

                ##################################################################
                # IMPLEMENT WORD-WISE AND LETTER-WISE ACCURACY HERE
                ##################################################################
                print('Testing...')
                accuracy, y_pred = crf.test(test_X, test_Y)
                print("C: {}, lr: {}, batch_size: {}, letter accuracy --->{}, word accuracy--->{}".format(C, learning_rate, batch_size, accuracy[0], accuracy[1]))

            step += 1
            if step > max_iters: raise StopIteration

            # sys.exit()

        del train, test
