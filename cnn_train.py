import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

import numpy as np
import argparse
import matplotlib.pyplot as plt

from data_loader import get_dataset
from AlexNet import AlexNet
from LeNet import LeNet

parser = argparse.ArgumentParser(description="Run AlexNet on OCR data")
parser.add_argument('-b' ,'--batch_size', type=int, required=False, help='Default=256. Size of each batch input for each step of optimization')
parser.add_argument('-e', '--num_epochs', type=int, required=False, help='Default=100. Number of times optimization step is run')
parser.add_argument('-m', '--model', required=True, help='Which kind of model to use. Can choose between "alexnet" or "lenet"')
parser.add_argument('-p', '--path', required=False, help='Determines the file name for the saved model')
args = parser.parse_args()

if args.model not in ['alexnet', 'lenet']:
	print("Error, model must either be 'alexnet' or 'lenet'. Exiting...")
	exit(-1)

# Filter the data into DataLoader objects
if args.batch_size is not None:
	BATCH_SIZE = args.batch_size
else: 
	BATCH_SIZE = 256

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("Using cuda to accelerate computations.")

if args.path is not None:
	PATH = args.path
else:
	PATH = "./{}_adam.pt".format(args.model)


def get_default_train_loader():
    dataset = get_dataset()
    split = int(0.8 * len(dataset.data))
    train_data = dataset.data[:split]
    train_target = dataset.target[:split]

    # Convert dataset into torch tensors
    train = data_utils.TensorDataset(torch.tensor(train_data).float(), torch.tensor(train_target).long())

    train_loader = data_utils.DataLoader(train,  # dataset to load from
                                         batch_size=BATCH_SIZE,  # examples per batch (default: 1)
                                         shuffle=False,
                                         sampler=None,  # if a sampling method is specified, `shuffle` must be False
                                         num_workers=5,  # subprocesses to use for sampling
                                         pin_memory=False,  # whether to return an item pinned to GPU
                                         )

    return train_loader


def get_default_test_loader():
    dataset = get_dataset()
    split = int(0.8 * len(dataset.data)) # train-test split
    test_data = dataset.data[split:]
    test_target = dataset.target[split:]

    # Convert dataset into torch tensors
    test = data_utils.TensorDataset(torch.tensor(test_data).float(), torch.tensor(test_target).long())

    test_loader = data_utils.DataLoader(test,  # dataset to load from
	                                batch_size=BATCH_SIZE,  # examples per batch (default: 1)
	                                shuffle=False,
	                                sampler=None,  # if a sampling method is specified, `shuffle` must be False
	                                num_workers=5,  # subprocesses to use for sampling
	                                pin_memory=False)  # whether to return an item pinned to GPU
    return test_loader

def process_data(dataset):
	# train-test split. Since we are using deep learning, we need a heavier split
	# in favor of training data
	split = int(0.80 * len(dataset.data)) 
	train_data, test_data = dataset.data[:split], dataset.data[split:]
	train_target, test_target = dataset.target[:split], dataset.target[split:]
	
	# Unwrap the word vectors such that the training and test data are all letters
	train_data_unwrapped = np.reshape(train_data, (-1, 16, 8))
	train_target_unwrapped = np.reshape(train_target, (-1, 26))

	test_data_unwrapped = np.reshape(test_data, (-1, 16, 8))
	test_target_unwrapped = np.reshape(test_target, (-1, 26))

	# Extract the indices of all the elements in the dataset that actually correspond to letters
	train_letters_indices = list(map(lambda letter: not np.array_equal(letter, np.zeros((26,))), train_target_unwrapped))
	test_letters_indices = list(map(lambda letter: not np.array_equal(letter, np.zeros((26,))), test_target_unwrapped))

	# Select out all of the training and test data that are not empty
	train_X = train_data_unwrapped[train_letters_indices] 
	train_Y = train_target_unwrapped[train_letters_indices]

	test_X = test_data_unwrapped[test_letters_indices]
	test_Y = test_target_unwrapped[test_letters_indices]

	train_X = np.expand_dims(train_X, axis=1)
	test_X = np.expand_dims(test_X, axis=1)
	
	#print("Train X Shape:", train_X.shape)
	#print("Test X Shape:", test_X.shape)

	return (train_X, train_Y), (test_X, test_Y) 


def unwrap_data(dataset):
	# train-test split. Since we are using deep learning, we need a heavier split
	# in favor of training data
	split = int(0.80 * len(dataset.data)) 
	train_data, test_data = dataset.data[:split], dataset.data[split:]
	train_target, test_target = dataset.target[:split], dataset.target[split:]
	
	# Unwrap the word vectors such that the training and test data are all letters
	train_data_unwrapped = np.reshape(train_data, (-1, 16, 8))
	train_target_unwrapped = np.reshape(train_target, (-1, 26))

	test_data_unwrapped = np.reshape(test_data, (-1, 16, 8))
	test_target_unwrapped = np.reshape(test_target, (-1, 26))

	return (train_X, train_Y), (test_X, test_Y) 


def letter_accuracy(loader, model):
	# Calculate the accuracy of the model with regards to either the training or the test set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            test_labels = torch.max(labels, 1)[1]

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == test_labels).sum().item()
    return 100 * correct / total


def encode_one_hot(predicted_labels, curr_batch_size):
    encoding = np.zeros((curr_batch_size * 14, 26))
    encoding[np.arange(curr_batch_size * 14), predicted_labels] = 1
    return encoding.reshape(curr_batch_size, 14, 26)
    

def remove_null_labels(letter_labels, predicted_labels):
    labels_indices = list(map(lambda letter: not np.array_equal(letter, np.zeros((26,))), letter_labels))
    clean_labels = letter_labels[labels_indices]
    clean_predictions = predicted_labels[labels_indices]
    return clean_labels, clean_predictions


def compare_word(labels, predicted):
    return np.array_equal(np.argmax(labels, 1), np.argmax(predicted, 1))


def word_accuracy(loader, model):
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in loader:
            images, labels = data
            curr_batch_size = labels.size(0)
            total += curr_batch_size
            
            # Unwrap the words such that they are just letters
            images = images.view(-1, 1, 16, 8)
            labels = labels.view(-1, 26)
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # Wrap the letters back up into words
            labels = labels.view(curr_batch_size, 14, 26)

            # Freeze the labels and predicted tensors to the cpu to use numpy to calculate word accuracy
            labels, predicted = labels.cpu().numpy(), predicted.cpu().numpy()

            # Encode the predictions into a one-hot encoded vector
            predicted_one_hot = encode_one_hot(predicted, curr_batch_size)

            # Remove the excess indices from the labels and the predicted_one_hot arrays
            clean_labels, clean_predictions = zip(*[remove_null_labels(label, prediction) 
                                                    for label, prediction in zip(labels, predicted_one_hot)])

            correct += np.sum([np.array_equal(label, prediction) for label, prediction in zip(clean_labels, clean_predictions)])
            
    return correct / total


def save_accuracies(training_accuracies, test_accuracies, metric, model, optimizer):
	tr_acc = np.array(training_accuracies)
	te_acc = np.array(test_accuracies)

	np.savetxt("./{}_{}_{}_training_accuracy.txt".format(model, optimizer, metric), tr_acc, delimiter=",")
	np.savetxt("./{}_{}_{}_test_accuracy.txt".format(model, optimizer, metric), te_acc, delimiter=",")


def main():
	print("Loading data...\n")

	dataset = get_dataset()
	(train_X, train_Y), (test_X, test_Y) = process_data(dataset)

	# Convert the dataset into torch tensors
	train = data_utils.TensorDataset(torch.tensor(train_X).float(), torch.tensor(train_Y).long())
	test = data_utils.TensorDataset(torch.tensor(test_X).float(), torch.tensor(test_Y).long())
	
	train_loader = data_utils.DataLoader(train,
	                                     batch_size=BATCH_SIZE,
	                                     shuffle=True,
	                                     num_workers=5,
	                                     sampler=None,
	                                     pin_memory=False)

	test_loader = data_utils.DataLoader(test,  # dataset to load from
	                                batch_size=BATCH_SIZE,  # examples per batch (default: 1)
	                                shuffle=False,
	                                sampler=None,  # if a sampling method is specified, `shuffle` must be False
	                                num_workers=5,  # subprocesses to use for sampling
	                                pin_memory=False)  # whether to return an item pinned to GPU
	
	if args.model == "lenet":
		print("Running LeNet on OCR")
		model = LeNet()
	else:
		print("Running AlexNet on OCR")
		model = AlexNet(num_classes=26)
	
	model.to(device)

	criterion = nn.CrossEntropyLoss()
	
	optimizer = optim.Adam(model.parameters(), lr=0.001)

	if args.num_epochs is not None:
		NUM_EPOCHS = args.num_epochs
	else:
		NUM_EPOCHS = 100

	print("Starting Training...\n")

	letter_training_accuracies = []
	letter_test_accuracies = []
	word_training_accuracies = []
	word_test_accuracies = []

	for epoch in range(NUM_EPOCHS):
		print("Processing epoch {}".format(epoch + 1))
		running_loss = 0.0

		for i_batch, sample in enumerate(train_loader, 0):
			train_X = sample[0]
			train_Y = sample[1]
			train_X, train_Y = train_X.to(device), train_Y.to(device)
			train_Y_labels = torch.max(train_Y, 1)[1]

			# # Zero the parameter gradients
			optimizer.zero_grad()

			# Run batch through model
			outputs = model(train_X)
			outputs.to(device)

			# Calculate loss
			tr_loss = criterion(outputs, train_Y_labels)
			tr_loss.backward()
			
			# Perform optimization step
			optimizer.step()

			running_loss += tr_loss.item()

			if i_batch % 20 == 0:
				print("Loss at Epoch {}, Batch {}: {}".format(epoch + 1, i_batch, running_loss / 5))
				running_loss = 0.0

		# Calculate the letter-level accuracy on the training and the test set
		letter_training_accuracy = letter_accuracy(train_loader, model)
		letter_test_accuracy = letter_accuracy(test_loader, model)
		letter_training_accuracies.append(letter_training_accuracy)
		letter_test_accuracies.append(letter_test_accuracy)

		# Calculate the word-level accuracy on the training and the test ser
		default_train_loader = get_default_train_loader()
		default_test_loader = get_default_test_loader()

		word_training_accuracy = word_accuracy(default_train_loader, model)
		word_test_accuracy = word_accuracy(default_test_loader, model)
		word_training_accuracies.append(word_training_accuracy)
		word_test_accuracies.append(word_test_accuracy)

		print('\nLetter Training Accuracy on epoch {}: {}'.format(epoch + 1, letter_training_accuracy))
		print('Letter Test Accuracy on epoch {}: {}'.format(epoch + 1, letter_test_accuracy))
		print('Word Training Accuracy on epoch {}: {}'.format(epoch + 1, word_training_accuracy))
		print('Word Training Accuracy on epoch {}: {}\n'.format(epoch + 1, word_test_accuracy))

	final_letter_test_accuracy = letter_accuracy(test_loader, model)
	final_word_test_accuracy = word_accuracy(default_test_loader, model)

	print("Letter Test accuracy of {} on OCR Data: {}".format(args.model, final_letter_test_accuracy))
	print("Word Test accuracy of {} on OCR Data: {}".format(args.model, final_word_test_accuracy))

	save_accuracies(letter_training_accuracies, letter_test_accuracies, "letter", args.model, "adam")
	save_accuracies(word_training_accuracies, word_test_accuracies, "word", args.model, "adam")

	# Save the model
	print("Saving {} model to {}".format(args.model, PATH))
	torch.save(model, PATH)


if __name__ == "__main__":
	main()