import torch
import numpy as np
import torch.utils.data as data_utils
from data_loader import get_dataset
import sys, math, time


def Messages(X, y, W, T):
    num_ex = len(X)
    num_features = 128
    num_label = 26

    l = torch.zeros((num_ex, num_label, 100))
    r = torch.zeros((num_ex, num_label, 100))

    for ex in range(num_ex):
        word = X[ex]
        word_label = y[ex]
        num_letter = len(word_label)

        def letter_func(l):
            return l.matmul(W)

        score = torch.stack([letter_func(l) for i, l in enumerate(torch.unbind(word, dim=0), 0)], dim=1)

        l[ex, :, 0] = 0
        r[ex, :, num_letter - 1] = 0
        for i in range(1, num_letter):
            v = l[ex, :, i - 1].add(score[:, i - 1]).view(num_label, 1)
            temp = T.add(v.repeat(1, num_label))
            max_temp = torch.max(temp, dim=0)[0].view((1, num_label))
            l[ex, :, i] = torch.add(
                torch.log(torch.sum(torch.exp((temp - max_temp.repeat(num_label, 1))), dim=0)), max_temp)

        for i in range(num_letter - 2, -1, -1):
            v = r[ex, :, i + 1].add(score[:, i + 1]).view(num_label, 1)
            temp = T.add(v.t().repeat(num_label, 1))
            max_temp = torch.max(temp, dim=1)[0].view(num_label, 1)
            r[ex, :, i] = torch.add(
                torch.log(torch.sum(torch.exp((temp - max_temp.repeat(1, num_label))), dim=1).view(26, 1)),
                max_temp).t()
    return l, r


def compute_log_probability(X, y, W, T, l, r):
    num_ex = len(X)
    f = 0
    for ex in range(num_ex):
        word = X[ex]
        word_label = y[ex]
        num_letter = len(word_label)

        def letter_func(l):
            return l.matmul(W)

        score = torch.stack([letter_func(l) for i, l in enumerate(torch.unbind(word, dim=0), 0)], dim=1)

        l_plus_score = torch.add(l[ex, :, :num_letter], score)
        r_plus_score = torch.add(r[ex, :, :num_letter], score)
        marg = torch.add(l_plus_score, r_plus_score)
        marg = marg - score
        t = torch.max(marg[:, 0])
        f = f - math.log(torch.sum(torch.exp((marg[:, 0] - t).float()))) - t
        for i in range(num_letter):
            lab = word_label[i]
            f = f + score[lab][i]
            if i < num_letter - 1:
                next_lab = word_label[i + 1]
                f = f + T[lab][next_lab]
    return f / num_ex


def ComputeGrad(X, y, W, T, l, r):
    num_ex = len(X)
    g_W = torch.zeros(W.shape)
    g_T = torch.zeros(T.shape)
    for ex in range(num_ex):
        word = X[ex]
        word_label = y[ex]
        num_letter = len(word_label)

        def letter_func(l):
            #             print(np.matmul(l,W))
            return l.matmul(W)

            # exp(<w.x>)

        score = torch.stack([letter_func(l) for i, l in enumerate(torch.unbind(word, dim=0), 0)], dim=1)

        l_plus_score = torch.add(l[ex, :, :num_letter], score)
        r_plus_score = torch.add(r[ex, :, :num_letter], score)
        marg = torch.add(l_plus_score, r_plus_score)
        marg = marg - score  # because score is added twice
        marg = torch.exp((marg - torch.max(marg, dim=0)[0].repeat(num_label, 1)).float())
        marg = marg / torch.sum(marg, dim=0).repeat(num_label, 1)  # Normalization

        for i in range(num_letter):
            lab = word_label[i]
            V = marg[:, i].view(num_label, 1)
            V[lab] = V[lab] - 1
            g_W = g_W - torch.matmul(word[i].view(num_features, 1), V.t())
            if i < num_letter - 1:
                next_lab = word_label[i + 1]
                V = torch.add(T, l_plus_score[:, i].view(num_label, 1).repeat(1, num_label))
                V = torch.add(V, r_plus_score[:, i + 1].view(num_label, 1).t().repeat(num_label, 1))
                V = torch.exp((V - torch.max(V)).float())
                g_T = g_T - V / torch.sum(V)
                g_T[lab][next_lab] = g_T[lab][next_lab] + 1
    grad = torch.cat((g_W.view(-1), g_T.view(-1)))
    return grad / num_ex


def loss(logprob, W, T, C, N):
    if C > 0:
        obj = W.norm() ** 2 / 2 + T.norm() ** 2 / 2 - logprob * C / N
    else:
        obj = logprob / N
    return obj


batch_size = 8

# Model parameters
num_features = 128
num_label = 26

# Fetch dataset
dataset = get_dataset()

split = int(0.5 * len(dataset.data)) # train-test split
train_data, test_data = dataset.data[:split], dataset.data[split:]
train_target, test_target = dataset.target[:split], dataset.target[split:]

# Convert dataset into torch tensors
train = data_utils.TensorDataset(torch.tensor(train_data).float(), torch.tensor(train_target).long())
test = data_utils.TensorDataset(torch.tensor(test_data).float(), torch.tensor(test_target).long())

X = train[:][0]
y = train[:][1]
y = [torch.nonzero(y[i])[:,1] for i in range(len(y))]
X = [X[i][:len(y[i])] for i in range(len(X))]
# print(X[0].shape)
# print(X[1].shape)

# x = torch.zeros((num_features*num_label + num_label*num_label), dtype=torch.double)
W = torch.zeros(num_features, num_label)
T = torch.zeros(num_label, num_label)

s = time.time()
f, b = Messages(X, y, W, T)
logp = compute_log_probability(X, y, W, T, f, b)
print(logp)
g = ComputeGrad(X, y, W, T, f, b)
print(g)
print(time.time()-s)
loss = loss(logp, W, T, 0, len(X))
print(loss)