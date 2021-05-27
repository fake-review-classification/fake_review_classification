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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            yield x[i + self.batch_size:], y[i + self.batch_size:], batch + 2
        if batch == 0:
            yield x, y, 1

    def train(self, x_train, y_train, x_val, y_val, fold=None, patience=None):
        '''to train model'''
        print('train model ...')
        train_losses, val_losses = [], []
        best_val_loss = np.inf
        count = 0
        
        if fold != None:
            x_train_split = list(torch.chunk(x_train, 5, dim=0))
            y_train_split = list(torch.chunk(y_train, 5, dim=0))

            copy_x_train_split = x_train_split.copy()
            left_x_train, x_val, right_x_train = copy_x_train_split[:fold], copy_x_train_split[fold], copy_x_train_split[fold+1:]
            x_train = torch.cat(left_x_train + right_x_train, dim=0)

            copy_y_train_split = y_train_split.copy()
            left_y_val, y_val, right_y_val = copy_y_train_split[:fold], copy_y_train_split[fold], copy_y_train_split[fold+1:]
            y_train = torch.cat(left_y_val + right_y_val, dim=0)
            
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

            print(f'Epoch {epoch+1} Train loss: {train_losses[-1]:.2f}. Validation loss: {val_losses[-1]:.2f}. Elapsed time: {elapsed:.2f}.')

            if patience != None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print('loss decresed {best_val_loss} -> {val_loss} save model...\n')
                    torch.save(self.model.state_dict(), '../checkpoint.pt')
                    count = 0

                if count >= patience:
                    print('Earlystoppiong!')
                    
                    plt.plot(train_losses, label="Training loss")
                    plt.plot(val_losses, label="Validation loss")
                    plt.legend()
                    plt.title("Losses")
                    plt.show()

                    return
                        
        plt.plot(train_losses, label="Training loss")
        plt.plot(val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        
        return
    
    def test(self, x_test, y_test):
        print('test model ...')
        self.model.load_state_dict(torch.load('../checkpoint.pt')) # load model for test
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
    
        # auc_score = roc_auc_score(y_test_np, y_preds_np, average=None)
        
        # return auc_score

    # def kfold_train_test(self, x_train, y_train, x_test, y_test, model_hyperparameter, k_fold=5):
    #     '''to train and test model with 5 fold'''
        
    #     print('train model ...')
        
    #     embed_num = model_hyperparameter['embed_num']
    #     embed_dim = model_hyperparameter['embed_dim']
    #     class_num = model_hyperparameter['class_num']
    #     kernel_num = model_hyperparameter['kernel_num']
    #     kernel_sizes = model_hyperparameter['kernel_sizes']
    #     dropout = model_hyperparameter['dropout']
    #     static = model_hyperparameter['static']
        
    #     auc_score_list = []
        
    #     x_train_split = list(torch.chunk(x_train, 5, dim=0))
    #     y_train_split = list(torch.chunk(y_train, 5, dim=0))
        
    #     for fold in range(k_fold):
    #         self.model = KimCNN(
    #                     embed_num=embed_num,
    #                     embed_dim=embed_dim,
    #                     class_num=class_num,
    #                     kernel_num=kernel_num,
    #                     kernel_sizes=kernel_sizes,
    #                     dropout=dropout,
    #                     static=static,
    #                     )
    #         self.model = self.model.to(device)
            
    #         copy_x_train_split = x_train_split.copy()
    #         left_x_train, fold_x_val, right_x_train = copy_x_train_split[:fold], copy_x_train_split[fold], copy_x_train_split[fold+1:]
    #         fold_x_train = torch.cat(left_x_train + right_x_train, dim=0)

    #         copy_y_train_split = y_train_split.copy()
    #         left_y_val, fold_y_val, right_y_val = copy_y_train_split[:fold], copy_y_train_split[fold], copy_y_train_split[fold+1:]
    #         fold_y_train = torch.cat(left_y_val + right_y_val, dim=0)
            
    #         train_losses, val_losses = [], []
            
    #         for epoch in range(self.n_epochs):
    #             start_time = time.time()
    #             train_loss = 0

    #             self.model.train(True)
    #             for x_batch, y_batch, batch in self.generate_batch_data(fold_x_train, fold_y_train):
    #                 y_pred = self.model(x_batch)
    #                 self.optimizer.zero_grad()
    #                 y_pred = y_pred.cpu()
    #                 loss = self.loss_function(y_pred, y_batch)

    #                 loss.backward()
    #                 self.optimizer.step()
    #                 train_loss += loss.item()

    #             train_loss /= batch
    #             train_losses.append(train_loss)
    #             elapsed = time.time() - start_time

    #             self.model.eval()  # disable dropout for deterministic output
    #             with torch.no_grad():  # deactivate autograd engine to reduce memory usage and speed up computations
    #                 val_loss, batch = 0, 1
    #                 for x_batch, y_batch, batch in self.generate_batch_data(fold_x_val, fold_y_val):
    #                     y_pred = self.model(x_batch)
    #                     y_pred = y_pred.cpu()
    #                     loss = self.loss_function(y_pred, y_batch)
    #                     val_loss += loss.item()
    #                 val_loss /= batch
    #                 val_losses.append(val_loss)

    #             print(
    #                 f'Epoch {epoch} Train loss: {train_losses[-1]:.2f}. Validation loss: {val_losses[-1]:.2f}. Elapsed time: {elapsed:.2f}.')

    #         plt.plot(train_losses, label="Training loss")
    #         plt.plot(val_losses, label="Validation loss")
    #         plt.legend()
    #         plt.title("Losses")
    #         plt.show()
    
    #         print('test model ...')

    #         self.model.eval()  # disable dropout for deterministic output
    #         with torch.no_grad():  # deactivate autograd engine to reduce memory usage and speed up computations
    #             y_preds = []
    #             batch = 0
    #             cnt = 0
    #             for x_batch, y_batch, batch in self.generate_batch_data(x_test, y_test):
    #                 y_pred = self.model(x_batch)
    #                 y_preds.extend(y_pred.cpu().numpy().tolist())
    #             y_preds_np = np.array(y_preds)

    #         y_test_np = np.array(y_test)

    #         auc_score = roc_auc_score(y_test_np, y_preds_np, average=None)

    #         print(f'kfold: {fold+1}/{k_fold},\tauc ascores : {auc_score}')
            
    #         auc_score_list.append(auc_score)
        
    #     return auc_score_list