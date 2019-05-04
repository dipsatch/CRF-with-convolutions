#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import string
import itertools
import scipy.optimize as opt
import time

def read_data(filename):
    X = []
    y = []
    
    with open(filename, 'r') as file:
        new_word = []
        new_word_letters = []
        line = file.readline().split()
        
        first_row_pixels = list(map(float, line[5:]))
        first_row_letter = ord(line[1])-97
        
        new_word.append(first_row_pixels)
        new_word_letters.append(first_row_letter)
        
        while True:
            line = file.readline().split()
            
            if line == []:
                break
            
            p_i_j = list(map(float, line[5:]))
            y_label = ord(line[1])-97
            next_char_index = int(line[2])
            char_index = int(line[0])
    
            new_word.append(p_i_j)
            new_word_letters.append(y_label)
        
            # If we reach the end of a word, enter the word into the X_train list
            if next_char_index == -1:
                X.append(np.array(new_word))
                y.append(np.array(new_word_letters))
                new_word = []
                new_word_letters = []
    
    return np.array(X), np.array(y)

def read_model():
    n = 26
    dim = 128
    T_i = 26
    T_j = 26
    model = np.loadtxt('model.txt')
    W = model[: n*dim].reshape(n, dim)
    T = model[n*dim: ].reshape(T_i, T_j)
    
    return W, T

def forward_messages(X, W, T):
    n = 26      # 26 letters
    m = len(X)  # word length
    f_msgs = np.zeros((m, n), dtype="float64") 
    
    # Implement the code from both of the comments in section 5.1 of the appendix
    for i in range(n):
        f_msgs[0, i] = np.exp(np.dot(X[0], W[i].T))
    for i in range(1, m):
        for a in range(n):
            product = np.exp(np.dot(X[i], W[a].T))
            tmp = np.zeros(n)
            for b in range(n):
                tmp[b] = np.exp(T[b, a]) * f_msgs[i - 1, b]
            f_msgs[i, a] = product * np.sum(tmp)
    return f_msgs

def backward_messages(X, W, T):
    n = 26      # 26 letters
    m = len(X)  # word length
    b_msgs = np.zeros((m, n), dtype="float64")
    for i in range(n):
        b_msgs[-1, i] = np.exp(np.dot(X[-1], W[i].T))
    for i in range(m - 2, -1, -1):
        for a in range(n):
            product = np.exp(np.dot(X[i], W[a].T))
            tmp = np.zeros(n)
            for b in range(n):
                tmp[b] = np.exp(T[b,a]) * b_msgs[i + 1, b]
            b_msgs[i, a] = product * np.sum(tmp)
    return b_msgs

def compute_log_probability(X, y, W, T):
    m = len(X)
    Z = np.sum(forward_messages(X, W, T)[-1])
    unnormalized = np.sum([np.dot(X[s], W[y[s]]) for s in range(m)])
    unnormalized += np.sum(np.array([T[y[i], y[i + 1]] for i in range(len(y) - 1)]))
    return np.log(np.exp(unnormalized) / Z)

def compute_grad_W(X, y, W, T):
    n = 26
    m = len(X)
    grad_W = np.empty((n, 128), dtype="float64")    
    f_msgs = forward_messages(X, W, T)
    b_msgs = backward_messages(X, W, T)
    Z = np.sum(f_msgs[-1])
    
    for i in range(n):  
        tmp = X[0] * b_msgs[0, i] + X[-1] * f_msgs[-1, i]
        for j in range(1, m-1):
            tmp += X[j] * (f_msgs[j, i] * b_msgs[j, i] / np.exp(np.dot(W[i], X[j])))
        grad_W[i] = sum(X[s] for s in range(m) if y[s] == i) - tmp / Z
    
    return np.array(grad_W)

def compute_grad_T(X, y, W, T):
    n = 26
    m = len(X)
    grad_T = np.zeros(T.shape)
    f_msgs = forward_messages(X, W, T)
    b_msgs = backward_messages(X, W, T)
    Z = np.sum(f_msgs[-1])

    for i in range(1, m):  
        for j in range(n):
            for k in range(n):
                grad_T[j, k] -= f_msgs[i-1, j] * b_msgs[i, k] * np.exp(T[j, k])

    grad_T = grad_T / Z

    for i in range(1, m):
        grad_T[y[i-1], y[i]] += 1

    return np.array(grad_T)

def compute_gradient(X, y, W, T):
    delta_W = compute_grad_W(X, y, W, T)
    delta_T = compute_grad_T(X, y, W, T)    
    return delta_W, delta_T

def write_gradient(grad_theta):
    with open('gradient.txt', 'w') as f:
        for grad in grad_theta:
            f.write(str(grad) + "\n") 


def get_crf_obj(X_train, y_train, W, T, c):    
    average_log_loss = -c * np.sum([compute_log_probability(X, y, W, T) for X, y in zip(X_train, y_train)]) / len(X_train)    
    W_norm = np.sum([(np.linalg.norm(Wy)) ** 2 for Wy in W]) / 2
    T_norm = np.sum([np.sum([Tij for Tij in row]) for row in T]) / 2
    return average_log_loss + W_norm + T_norm

def crf_obj(theta, X_train, y_train, c):
    #print("iteration time---->{}".format(time.time()-start)) 
    
    # x is a vector as required by the solver. So reshape it to w_y and T
    W = np.reshape(theta[:26*128], (26, 128))  # each column of W is w_y (128 dim)
    T = np.reshape(theta[128*26:], (26, 26))  # T is 26*26
    
    f = get_crf_obj(X_train, y_train, W, T, c)                                                                                                   
    return f

def crf_obj_prime(theta, X_data, y_data, c):
    W = np.reshape(theta[:128*26], (26, 128))  # each column of W is w_y (128 dim)
    T = np.reshape(theta[128*26:], (26, 26))  # T is 26*26
    n = 26

    grad_Ws = np.zeros((n, X_data[0].shape[1]), "float64")
    grad_Ts = np.zeros((n, n), "float64")

    for X, y in zip(X_data, y_data):
        grad_Ws += compute_grad_W(X, y, W, T)
        grad_Ts += compute_grad_T(X, y, W, T)
    
    grad_Ws = (-c) * grad_Ws / float(len(X_data))
    grad_Ts = (-c) * grad_Ts / float(len(X_data))

    grad_theta = np.concatenate([grad_Ws.reshape(-1), grad_Ts.reshape(-1)])
    
    return grad_theta

def crf_test(x, X_test, y_test):
    
    # x is a vector. so reshape it into w_y and T
    W = np.reshape(x[:128*26], (26, 128))  # each column of W is w_y (128 dim)
    T = np.reshape(x[128*26:], (26, 26))  # T is 26*26

    # Compute the CRF prediction of test data using W and T
    y_predict = crf_decode(W, T, X_test)

    # Compute the test accuracy by comparing the prediction with the ground truth
    accuracy = compare(y_predict, y_test)
    return accuracy, y_predict

def compare(y_predict, y_train):
    totalletter,letter_count,word_accuracy=0,0,0
    for y_p,y in zip(y_predict, y_train):
        letter_accuracy=sum([1 for i,j in zip(y_p,y) if i==j ])
        totalletter+=letter_accuracy
        if letter_accuracy==len(y):word_accuracy+=1
        letter_count+=len(y_p)
    return float(totalletter/letter_count),float(word_accuracy/len(y_predict))        

def crf_decode(W, T, X_train):
    y_pred=[]
    
    def maxSumBottomUp(X, W, T):
        n = 26      # 26 letters
        m = len(X)  # word length
        l = np.zeros((m, n))
        opts = np.zeros((m, n))
        yStar = np.zeros(m, dtype=np.int8)
        for i in range(1, m):
            for a in range(n):
                tmp = np.zeros(n)
                for b in range(n):
                    tmp[b] = np.dot(X[i-1], W[b]) + T[a, b] + l[i-1, b]
                l[i, a] = max(tmp)
        for b in range(n):
            opts[m-1, b] = np.dot(X[m-1], W[b]) + l[m-1, b]
        MAP = max(opts[m-1])
        yStar[m-1] = np.argmax(opts[m-1])
        for i in range(m-1, 0, -1):
            for b in range(n):
                opts[i-1, b] = np.dot(X[i-1], W[b]) + T[yStar[i], b] + l[i-1, b]
            yStar[i-1] = np.argmax(opts[i-1])

        return MAP, yStar

    for X in X_train:
        y_pred.append(maxSumBottomUp(X, W, T)[1])
    return y_pred
        
def main():
    C=1000
    X_train, y_train = read_data('train.txt')
    X_test, y_test = read_data('test.txt')
    W, T = read_model()
    theta = np.concatenate([W.reshape(-1), T.reshape(-1)]) 
    print("Calculating gradients and saving to gradient.txt ...")
    gradient_theta = crf_obj_prime(theta, X_train, y_train, C)
    write_gradient(gradient_theta.reshape(-1))

    start=time.time()

    print('Training CRF ... c = {} \n'.format(C))
    x0 = np.zeros((26*128+26**2,1))
    theta, fmin, _ = opt.fmin_l_bfgs_b(crf_obj, x0, fprime=crf_obj_prime,args=(X_train, y_train, C), disp=1, maxiter=100)# result = opt.fmin_tnc(crf_obj, x0,fprime=None, args = [X_train[:100], y_train[:100], C], maxfun=100,ftol=1e-3, disp=5)         
    print("training time---->{}".format(time.time()-start))   


    with open('solution.txt','w') as f:
        for grad in theta:
            f.write("%s\n" % grad) 
            
    accuracy,y_predict = crf_test(theta, X_test, y_test)
    
    print("C: {} letter accuracy --->{} word accuracy--->{}".format(C, accuracy[0],accuracy[1]))
    
    with open('prediction.txt','w') as f:
        for p_y in y_predict:
            for l in p_y:
                f.write("%s\n" % l)

if __name__ == "__main__":
    main()


# In[ ]:




