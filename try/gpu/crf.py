import torch
# from conv_dg import Conv
from conv_dg_vec import Conv


class CRF(torch.nn.Module):
    def __init__(self, input_dim, conv_layers, num_labels, batch_size, cuda):
        """
        Linear chain CRF as in Assignment 2
        """
        super(CRF, self).__init__()

        self.input_dim = input_dim
        self.crf_input_dim = None  # Calculated inside init_params() based on Conv layer I/O dims
        self.conv_layers = conv_layers
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.use_cuda = cuda
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.conv_objects = torch.nn.ModuleList()  # Stores multiple Conv objects
        self.Ks = []   # Stores multiple Conv filters
        self.W = None  # CRF parameter (node potential)
        self.T = None  # CRF parameter (transition weights between node pairs)

    def init_params(self, Ks=None, W=None, T=None):  # Define trainable parameters here, should be nn.Parameter or with requires_grad=True
        if Ks is None:    # Initialize default Conv filters if nothing is passed via arguments
            for conv_layer in self.conv_layers:
                self.Ks.append(torch.nn.Parameter(torch.randn(conv_layer['filter_shape'], dtype=torch.float).uniform_(-0.1, 0.1)))
        else:
            self.Ks = Ks  # Initialize Conv filters with trained filters if passed via arguments

        # Define input & output dims of each Conv layer & store inside self.conv_layers. Formula: OutputDim = 1 + (InputDim − F + 2P)/S
        last_output_dim = self.input_dim.copy()
        for i, conv_layer in enumerate(self.conv_layers):
            self.conv_objects.append(Conv(K=self.Ks[i], padding=conv_layer['padding'], stride=conv_layer['stride'], device=self.device))  # Instantiate a new Conv object & append in ModuleList to register
            # self.conv_objects.append(torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=conv_layer['filter_shape'][-1], padding=conv_layer['padding'], stride=conv_layer['stride']))  # Instantiate a new Conv object & append in ModuleList to register
            conv_layer['input_dim'] = last_output_dim.copy()
            output_dim_height = int(1 + (conv_layer['input_dim']['height'] - conv_layer['filter_shape'][-2] + 2*conv_layer['padding'])/conv_layer['stride'])
            output_dim_width  = int(1 + (conv_layer['input_dim']['width']  - conv_layer['filter_shape'][-1] + 2*conv_layer['padding'])/conv_layer['stride'])
            output_dim_flattened = output_dim_height * output_dim_width
            conv_layer['output_dim'] = {'flattened': output_dim_flattened, 'height': output_dim_height, 'width': output_dim_width}
            last_output_dim = conv_layer['output_dim'].copy()

        # Define W with the flattened output dim of the last Conv filter that is getting stored in self.crf_input_dim
        self.crf_input_dim = last_output_dim['flattened']
        self.W = torch.nn.Parameter(torch.zeros((self.crf_input_dim, self.num_labels), dtype=torch.float)) if W is None else W
        self.T = torch.nn.Parameter(torch.zeros((self.num_labels, self.num_labels), dtype=torch.float)) if T is None else T

        ### Use GPU if available
        if self.use_cuda:
            [m.cuda() for m in self.modules()]

    def get_conv_feats(self, X):
        """
        Generate convolution features for a given word
        """
        convfeatures = X
        for i, conv_layer in enumerate(self.conv_layers):
            # convfeatures = self.conv_objects[i](convfeatures.view(convfeatures.shape[:-1] + (conv_layer['input_dim']['height'], conv_layer['input_dim']['width'])))  # Extend the last dim of X, e.g. 128 to (16,8)
            # convfeatures = convfeatures.view(convfeatures.shape[:-2] + (convfeatures.shape[-1] * convfeatures.shape[-2],))  # Flatten last 2 dims, e.g. (16,8) to 128

            convfeatures = self.conv_objects[i](convfeatures.view(convfeatures.shape[0] * convfeatures.shape[1], 1, conv_layer['input_dim']['height'],conv_layer['input_dim']['width']))  # Extend the last dim of X, e.g. 128 to (16,8)
            convfeatures = convfeatures.view(self.batch_size, int(convfeatures.shape[0] / self.batch_size), convfeatures.shape[-1] * convfeatures.shape[-2])  # Flatten last 2 dims, e.g. (16,8) to 128

            # convfeatures = torch.nn.functional.conv2d(convfeatures.view(convfeatures.shape[0] * convfeatures.shape[1], 1, conv_layer['input_dim']['height'],conv_layer['input_dim']['width']), self.Ks[i], padding=conv_layer['padding'], stride=conv_layer['stride'])
            # convfeatures = convfeatures.view(self.batch_size, int(convfeatures.shape[0] / self.batch_size), convfeatures.shape[-1] * convfeatures.shape[-2])

        return convfeatures

    def forward(self, X, labels):
        """
        Implement the objective of CRF here.
        The input (features) to the CRF module should be convolution features.
        """
        self.batch_size = len(X)
        features = self.get_conv_feats(X)
        log_prob = CRFautograd.apply(features, labels, self.W, self.T, self.device)
        return log_prob

    def loss(self, log_prob, C):
        """
        Compute the negative conditional log-likelihood of a labelling given a sequence.
        """
        loss = (self.W.norm() ** 2 / 2 + self.T.norm() ** 2 / 2 - log_prob * C) if C > 0 else log_prob
        return loss

    def test(self, X_test, y_test_1hot):
        y_pred = self.predict(X_test)
        y_test = torch.zeros(y_pred.shape, device=self.device)
        for i, y_t in enumerate(y_test_1hot):  # Converting 1-hot vectors to letter indices
            converted = torch.nonzero(y_t)[:, 1]
            y_test[i, :len(converted)] = converted
        accuracy = self.compare(y_pred, y_test)
        return accuracy, y_pred

    def compare(self, y_pred, y_test):
        totalletter, letter_count, word_accuracy = 0, 0, 0
        for y_p, y in zip(y_pred, y_test):
            letter_accuracy = sum([1 for i, j in zip(y_p, y) if i.item() == j.item()])
            totalletter += letter_accuracy
            if letter_accuracy == len(y): word_accuracy += 1
            letter_count += len(y_p)
        return float(totalletter / letter_count), float(word_accuracy / len(y_pred))

    def predict(self, X):  # returns predicted sequence
        features = self.get_conv_feats(X)
        prediction = self.crf_decode(features, self.W.t(), self.T)
        return prediction

    def crf_decode(self, X_train, W, T):
        y_pred = torch.zeros(X_train.shape[0], X_train.shape[1], device=self.device)

        def maxSumBottomUp(X, W, T):
            n = len(T)  # 26 letters
            m = torch.nonzero(X)[-1][0].item() + 1  # word length, without zero padding
            X = X[:m]   # Removing padding
            l = torch.zeros((m, n), device=self.device)
            opts = torch.zeros((m, n), device=self.device)
            yStar = torch.zeros(m, dtype=torch.int8, device=self.device)
            for i in range(1, m):
                for a in range(n):
                    tmp = torch.zeros(n, device=self.device)
                    for b in range(n):
                        tmp[b] = torch.dot(X[i - 1], W[b]) + T[a, b] + l[i - 1, b]
                    l[i, a] = max(tmp)
            for b in range(n):
                opts[m - 1, b] = torch.dot(X[m - 1], W[b]) + l[m - 1, b]
            MAP = torch.max(opts[m - 1])
            yStar[m - 1] = opts[m - 1].max(0)[1]
            for i in range(m - 1, 0, -1):
                for b in range(n):
                    opts[i - 1, b] = torch.dot(X[i - 1], W[b]) + T[yStar[i], b] + l[i - 1, b]
                yStar[i - 1] = opts[i - 1].max(0)[1]
            return MAP, yStar

        for i, X in enumerate(X_train):
            _, yStar = maxSumBottomUp(X, W, T)
            y_pred[i, :len(yStar)] = yStar
        return y_pred


class CRFautograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, y, W, T, device):
        l, r = CRFautograd.fwdBwdMsgs(X, y, W, T, device)
        ctx.save_for_backward(X, y, W, T, l, r)
        ctx.device = device
        return CRFautograd.compute_log_probability(X, y, W, T, l, r, device)

    @staticmethod
    def backward(ctx, grad_out):
        # print('Bwd called')
        X, y, W, T, l, r = ctx.saved_tensors
        device = ctx.device
        g_W, g_T, g_X = CRFautograd.compute_gradient(X, y, W, T, l, r, device)
        g_W = grad_out * g_W
        g_T = grad_out * g_T
        g_X = grad_out * g_X
        return g_X, None, g_W, g_T, None

    @staticmethod
    def fwdBwdMsgs(X, y, W, T, device):
        num_ex = len(X)
        num_label = len(T)
        max_word_length = X.shape[1]  # 14

        l = torch.zeros((num_ex, num_label, max_word_length), device=device)
        r = torch.zeros((num_ex, num_label, max_word_length), device=device)

        for ex in range(num_ex):
            word_label = torch.nonzero(y[ex])[:, 1]
            word = X[ex, :len(word_label)]
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

    @staticmethod
    def compute_log_probability(X, y, W, T, l, r, device):
        num_ex = len(X)
        f = 0
        for ex in range(num_ex):
            word_label = torch.nonzero(y[ex])[:, 1]
            word = X[ex, :len(word_label)]
            num_letter = len(word_label)

            def letter_func(l):
                return l.matmul(W)

            score = torch.stack([letter_func(l) for i, l in enumerate(torch.unbind(word, dim=0), 0)], dim=1)

            l_plus_score = torch.add(l[ex, :, :num_letter], score)
            r_plus_score = torch.add(r[ex, :, :num_letter], score)
            marg = torch.add(l_plus_score, r_plus_score)
            marg = marg - score
            t = torch.max(marg[:, 0])
            f = f - torch.log(torch.sum(torch.exp((marg[:, 0] - t).float()))) - t
            for i in range(num_letter):
                lab = word_label[i]
                f = f + score[lab][i]
                if i < num_letter - 1:
                    next_lab = word_label[i + 1]
                    f = f + T[lab][next_lab]
        return f / num_ex

    @staticmethod
    def compute_gradient(X, y, W, T, l, r, device):
        num_ex = len(X)
        num_features = len(W)
        num_label = len(T)  # 26
        max_word_length = X.shape[1]  # 14
        g_W = torch.zeros(W.shape, device=device)
        g_T = torch.zeros(T.shape, device=device)
        g_X = torch.zeros((num_ex, max_word_length, num_features), device=device)
        for ex in range(num_ex):
            word_label = torch.nonzero(y[ex])[:, 1]
            word = X[ex, :len(word_label)]
            num_letter = len(word_label)

            def letter_func(l):
                return l.matmul(W)

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
                V_X = V.clone()
                V[lab] = V[lab] - 1
                g_W = g_W - torch.matmul(word[i].view(num_features, 1), V.t())
                g_X[ex, i, :] = (W[:, lab].view(num_features, 1) - W.mm(V_X)).t()
                if i < num_letter - 1:
                    next_lab = word_label[i + 1]
                    V = torch.add(T, l_plus_score[:, i].view(num_label, 1).repeat(1, num_label))
                    V = torch.add(V, r_plus_score[:, i + 1].view(num_label, 1).t().repeat(num_label, 1))
                    V = torch.exp((V - torch.max(V)).float())
                    g_T = g_T - V / torch.sum(V)
                    g_T[lab][next_lab] = g_T[lab][next_lab] + 1

        return g_W / num_ex, g_T / num_ex, g_X / num_ex

