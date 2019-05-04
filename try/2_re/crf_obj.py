# still too-good-to-be-true(aka wrong) values, will fix; correct upto score calculation at least
import math
import numpy as np

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
            # char_index = int(line[0])
    
            new_word.append(p_i_j)
            new_word_letters.append(y_label)
        
            # If we reach the end of a word, enter the word into the X_train list
            if next_char_index == -1:
                X.append(np.array(new_word))
                y.append(np.array(new_word_letters))
                new_word = []
                new_word_letters = []
    
    return np.array(X), np.array(y)

def crf_obj(x, word_list, word_labels, C):
    num_ex=len(word_list)
    w=word_list[0]
    num_features = len(w[1]) #128
    num_label = int((math.sqrt(num_features**2+4*len(x)) - num_features)/2) #26
    # print(f'num_features= {num_features}, num_label= {num_label}')
    W = np.reshape(x[:num_features*num_label], (num_features, num_label), order='F')
    T = np.reshape(x[num_features*num_label:], (num_label, num_label), order='F')
    l = np.zeros((num_label, 100))
    r = np.zeros((num_label, 100))
    # r = l
    g_W = np.zeros(W.shape)
    g_T = np.zeros(T.shape)
    g_X = np.zeros((num_ex,num_features,100))
    f = 0

    for ex in range(num_ex):
        word = word_list[ex]
        word_label = word_labels[ex]
        num_letter = len(word)     
        # print(f'{word.shape}, {W.shape}')
        # print(f'{word.dtype}, {W.dtype}')
        # print(f'{word}')
        # score = np.transpose(np.matmul(word, W)) #26 x num_letters
        
        def letter_func(l):
            return np.matmul(l,W)
        score = np.transpose(np.apply_along_axis(letter_func, 1, word))
        # print(f'{score.shape}') 
        # if(ex==0):
        #     print(score)
        # exit()
        l[:,0] = 0
        r[:,num_letter-1] = 0
        for i in range(1, num_letter):
            v = np.reshape(np.add(l[:,i-1], score[:,i-1]), (num_label,1))
            temp = np.add(T, np.tile(v, (1,num_label)))
            max_temp = np.reshape(np.amax(temp, axis=0), (1,num_label))
            # print(f'{v.shape}, {temp.shape}, {max_temp.shape}') #(26, 1), (26, 26), (1, 26)
            # print(f'1: {np.add(np.log(np.sum(np.exp(np.subtract(temp, np.tile(max_temp, (num_label,1)))), axis=0)), max_temp).shape}, 2: {l[:, i].shape}') #1: (1, 26), 2: (26,)
            l[:, i] = np.add(np.log(np.sum(np.exp(np.subtract(temp, np.tile(max_temp, (num_label,1)))), axis=0)), max_temp)
            
        for i in range(num_letter-2, -1, -1):
            v = np.reshape(np.add(r[:,i+1], score[:,i+1]), (num_label,1))  
            temp = np.add(T, np.tile(np.transpose(v), (num_label,1)))
            max_temp = np.reshape(np.amax(temp, axis=1), (num_label,1)) 
            r[:, i] = np.transpose(np.add(np.log(np.reshape(np.sum(np.exp(np.subtract(temp, np.tile(max_temp, (1,num_label)))), axis=1), (26,1))), max_temp))
            
        #computing gradient
        l_plus_score = np.add(l[:, :num_letter], score)
        r_plus_score = np.add(r[:, :num_letter], score)     
        marg = np.add(l_plus_score, r_plus_score)
        marg = np.subtract(marg, score)
        t = np.amax(marg[:,0])
        f = f - math.log(np.sum(np.exp(marg[:,0]-t))) - t
        marg = np.exp(np.subtract(marg, np.tile(np.amax(marg, axis=0), (num_label, 1))))
        marg = np.divide(marg, np.tile(np.sum(marg, axis=0), (num_label, 1)))  

        for i in range(num_letter):
            lab = word_label[i]
            f = f + score[lab][i]
            V = np.reshape(marg[:,i], (num_label,1))  #26x1
            V_X = V
            V[lab] = V[lab] - 1
            g_W = np.subtract(g_W, np.matmul(np.reshape(word[i], (num_features,1)), np.transpose(V)))
            g_X[ex,:,i] = np.transpose(np.subtract(np.reshape(W[:,lab], (num_features,1)), np.matmul(W, V_X)))
            if i < num_letter-1:
                next_lab = word_label[i+1]
                f = f + T[lab][next_lab]
                V = np.add(T, np.tile(np.reshape(l_plus_score[:,i], (num_label,1)), (1, num_label)))
                V = np.add(V, np.tile(np.transpose(r_plus_score[:,i+1]), (num_label, 1)))
                V = np.exp(V - np.amax(V))
                g_T = g_T - V / np.sum(V)
                g_T[lab][next_lab] = g_T[lab][next_lab] + 1
    # print(g_X[3437,:1,:num_letter])
    if C>0:
        f = np.linalg.norm(x)**2/2 - f*C/num_ex
        # g = x - [g_W(:);g_T(:)] * (C / num_ex)
        g = np.concatenate((g_W.flatten('F'), g_T.flatten('F'))) * (C / num_ex)
        g = np.subtract(x, g)
    else:  
        f = f / num_ex
        # g = [g_W(:); g_T(:)] / num_ex
        g = np.concatenate((g_W.flatten('F'), g_T.flatten('F'))) / num_ex

    return f, g


X_train, y_train = read_data('data\\train.txt')
X_test, y_test = read_data('data\\test.txt')
with open('model.txt') as f:
    content=f.readlines()
x=[float(x.rstrip()) for x in content]
f,g=crf_obj(x,X_train,y_train,-1)
print(f'Obj = {f}')
with open('crf_g_py.txt', 'w') as f:
    for grad in g:
        f.write("%s\n" % grad)

