import time
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score

random_seed = 42

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


class KimCNN(nn.Module):
    '''CNN model'''

    def __init__(self, embed_num, embed_dim, class_num, kernel_num, kernel_sizes, dropout, static):
        super(KimCNN, self).__init__()

        V = embed_num
        D = embed_dim
        C = class_num
        Co = kernel_num
        Ks = kernel_sizes

        self.static = static
        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3)
             for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2)
             for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        output = self.sigmoid(logit)
        return output

class train_test_model:
    def __init__(self, model, n_epochs=10, batch_size=16, lr=0.001, optimizer=torch.optim.Adam, loss_fn=nn.BCELoss()):
        self.model = model
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = lr
        self.optimizer = optimizer(
            self.model.parameters(), lr=self.learning_rate)
        self.loss_function = loss_fn

    def generate_batch_data(self, x, y):
        '''To make batch size data'''
        i, batch = 0, 0
        for batch, i in enumerate(range(0, len(x) - self.batch_size, self.batch_size), 1):
            x_batch = x[i: i + self.batch_size]
            y_batch = y[i: i + self.batch_size]
            yield x_batch, y_batch, batch
        if i + self.batch_size < len(x):
            yield x[i + self.batch_size:], y[i + self.batch_size:], batch + 1
        if batch == 0:
            yield x, y, 1

    def train(self, x_train, y_train, x_val, y_val):
        'to train model'
        print('train model ...')
        train_losses, val_losses = [], []

        for epoch in range(self.n_epochs):
            start_time = time.time()
            train_loss = 0

            self.model.train(True)
            for x_batch, y_batch, batch in self.generate_batch_data(x_train, y_train):
                y_pred = self.model(x_batch)
                self.optimizer.zero_grad()
                y_pred = y_pred.cpu()
                loss = self.loss_function(y_pred, y_batch)

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= batch
            train_losses.append(train_loss)
            elapsed = time.time() - start_time

            self.model.eval()  # disable dropout for deterministic output
            with torch.no_grad():  # deactivate autograd engine to reduce memory usage and speed up computations
                val_loss, batch = 0, 1
                for x_batch, y_batch, batch in self.generate_batch_data(x_val, y_val):
                    y_pred = self.model(x_batch)
                    y_pred = y_pred.cpu()
                    loss = self.loss_function(y_pred, y_batch)
                    val_loss += loss.item()
                val_loss /= batch
                val_losses.append(val_loss)

            print(
                f'Epoch {epoch+1} Train loss: {train_losses[-1]:.2f}. Validation loss: {val_losses[-1]:.2f}. Elapsed time: {elapsed:.2f}.')

        plt.plot(train_losses, label="Training loss")
        plt.plot(val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
    
    def test(self, x_test, y_test):
        print('test model ...')

        self.model.eval()  # disable dropout for deterministic output
        with torch.no_grad():  # deactivate autograd engine to reduce memory usage and speed up computations
            y_preds = []
            batch = 0
            cnt = 0
            for x_batch, y_batch, batch in self.generate_batch_data(x_test, y_test):
                y_pred = self.model(x_batch)
                y_preds.extend(y_pred.cpu().numpy().tolist())
            y_preds_np = np.array(y_preds)

        y_test_np = np.array(y_test)
        
        return y_test_np, y_preds_np
    
#         auc_scores = roc_auc_score(y_test_np, y_preds_np, average=None)
        
#         return auc_scores
        
