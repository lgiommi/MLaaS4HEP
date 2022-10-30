import torch
import torch.nn as nn
import torch.nn.functional as fun
import os.path
from datetime import datetime

class ClassifierNN(nn.Module):

    def __init__(self, idim,
                 activation=fun.relu):
        super().__init__()

        self.last_save = None
        self.layout = (idim, 256, 128, 1)
        self.inference_mode = False  # training clients: change this attribute to False
        self.activation = activation
        self.layers = nn.ModuleList()
        for num_nodes, num_nodes_next in zip(self.layout[:-1], self.layout[1:]):
            self.layers.append(nn.Linear(num_nodes, num_nodes_next))

    def forward(self, x):
        for layer in self.layers[:-1]:
            if not isinstance(x, torch.Tensor):
                x = torch.Tensor(x)
            x = self.activation(layer(x))

        x = torch.sigmoid(self.layers[-1](x))
        return x

    def train(self, mode=True):
        super(ClassifierNN, self).train()
        self.inference_mode = False

    def eval(self):
        super(ClassifierNN, self).eval()
        self.inference_mode = True

    def train_clf(self, model, train_loader, val_loader):
        grace = 0
        max_epochs=10
        min_gain=0.01
        grace_limit=4
        learning_rate=0.002
        momentum=0.002
        weight_decay=1e-5
        save_dir='/weights'
        weight_file_tag=None
        train_history = []
        test_history = []
        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        model.train()

        weigh_path = None
        best_loss = 1.0
        for epoch in range(max_epochs):
            mean_train_loss = 0.0
            for i, (xs, ys) in enumerate(train_loader):
                optimizer.zero_grad() # reset gradients
                outputs = model(xs)
                train_loss = loss_function(outputs, ys)
                train_loss.backward() # gradient back propagation
                optimizer.step()
                mean_train_loss = (mean_train_loss * i + float(train_loss)) / (i + 1)
            train_history.append(mean_train_loss)

            mean_test_loss = 0.0
            for i, (xs, ys) in enumerate(val_loader):
                outputs = model(xs)
                eval_loss = loss_function(outputs, ys)
                mean_test_loss = (mean_test_loss * i + float(eval_loss)) / (i + 1)
            test_history.append(mean_test_loss)

            print('Epoch {}, mean train/test loss: {:.4f}/{:.4f}'.format(epoch+1, mean_train_loss, mean_test_loss))
            if (best_loss - mean_test_loss) / best_loss < min_gain:
                if grace == 0:
                    print('Entering grace period (limit {})'.format(grace_limit))
                    grace += 1
                elif grace < grace_limit:
                    grace += 1
                else:
                    print('Nothing more to learn. Training finished.')
                    break
            else:
                if grace > 0:
                    grace = 0
                    print('Survived grace period.')
                best_loss = mean_test_loss
        else:
            print('Maximum number of epochs ({}) reached. Training terminated.'.format(max_epochs))

        return train_history, test_history

def model(idim):
    """Simple pyTorch model for testing purpose"""
    torch_model = ClassifierNN(idim)
    return torch_model
