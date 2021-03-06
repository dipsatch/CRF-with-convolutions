Exploring Conditional Random Fields (CRFs), but we will use additional image level features such as convolutions to aid the training. We will use PyTorch to implement our CRF model and convolutions. This is an end-to-end machine learning solution to a problem in PyTorch. In the process, we will do differentiable layer-wise programming, which is common across all the popular deep learning frameworks that exist today.

Tasks done:

▪ Implemented a conditional random field model for Optical Character Recognition(OCR), with emphasis on inference and performance test.
▪ Benchmarking done by comparing CRF with multi-class linear SVM
▪ Used image level features such as convolutions to aid the training of CRF
▪ Comparison of LBFGS and ADAM optimizers
▪ Benchmarking with renowned networks such as VGG, AlexNet, ResNet and GoogLeNet
▪ Used PyTorch and utilized GPUs for training




3.
------------------
Files: conv_test.py, conv_dg.py
To test: run conv_test.py to get output of convolution operation as standalone test.
Input matrix X and filter matrix K used are as given in Assignment question.

4.
------------------
Files: crf.py, crf_test.py, conv_dg_vec.py
To test: run crf_test.py. Filter size parameters, padding and stride values can be changed in lines 20-25 of crf_test.py.



5.
------------------
In order to run AlexNet using either the Adam or lbfgs optimizers, run the respective commands:
	
	python3 cnn_train.py -b [BATCH_SIZE] -e [NUM_EPOCHS] -m [MODEL] -p [PATH]
	python3 cnn_train_lbfgs.py -b [BATCH_SIZE] -e [NUM_EPOCHS] -m [MODEL] -p [PATH]

The model argument is required to be either "alexnet" or "lenet" (lenet is the baseline to check performance against).

The batch size, num epoch, and path arguments are optional, and they're default values are:
	batch_size=256, num_epochs=100, path='/results/model/{}.pt' where {} stands for the model name.

Note: When running the AlexNet model with LBFGS, there might be some memory problems, so
in order to avoid those, try using a high batch size. I ran it with 512 for LBFGS, and 256 with Adam