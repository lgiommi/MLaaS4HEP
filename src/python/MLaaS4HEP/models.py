#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=R0913,R0914
"""
File       : models.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Models module defines wrappers to train MLaaS4HEP workflows.

User should provide a model implementation (so far in either in Keras or PyTorch),
see ex_keras.py and ex_pytorch.py, respectively for examples.

The Trainer class defines a thin wrapper around user model and provides uniformat
APIs to fit the model and yield predictions.
"""
from __future__ import print_function, division, absolute_import

# system modules
import time
import json
# numpy modules
import numpy as np
import pickle
import inspect

#sklearn modules
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

# keras modules
from tensorflow.keras.utils import to_categorical

# pytorch modules
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None

# MLaaS4HEP modules
from generator import RootDataGenerator, MetaDataGenerator, file_type
from utils import load_code

class Trainer(object):
    """
    Trainer class defines a think wrapper around given user model.
    It defines basic `fit` and `predict` APIs.
    """
    def __init__(self, model, verbose=0):
        self.model = model
        self.verbose = verbose
        self.cls_model = '{}'.format(type(self.model)).lower()
        if self.verbose:
            try:
                print(self.model.summary())
            except AttributeError:
                print(self.model)

    def fit(self, x_train, y_train, **kwds):
        """
        Fit API of the trainer.

        :param data: the ROOT IO data in form of numpy array of data and mask vectors.
        :param y_train: the true values vector for input data.
        :param kwds: defines input set of parameters for end-user model.
        """
        if self.verbose > 1:
            print("Perform fit on {} data with {}"\
                    .format(np.shape(x_train), kwds))
        if self.cls_model.find('keras') != -1:
            self.model.fit(x_train, y_train, verbose=self.verbose, **kwds)
            baseline_results = self.model.evaluate(x_train, y_train, verbose=0)
            print('\n')
            for name, value in zip(self.model.metrics_names, baseline_results):
                print(name, 'train: ', value)
            if hasattr(self.model, 'log_loss'):
                loss = self.model.metrics.log_loss(y_train, self.model.predict_proba(x_train)[:,1])
                print("Log Loss Function: {}".format(loss))

        elif self.cls_model.find('torch') != -1:
            if hasattr(self.model, 'train_clf'):
                self.model.train_clf(model=self.model, train_loader=x_train, val_loader=y_train)
            else:
                self.torch_train(self.model, train_loader=x_train, val_loader=y_train, **kwds)

        elif self.cls_model.find('xgboost') != -1:
            print('##### Fitting the model #####')
            self.model.fit(x_train, y_train)

        elif self.cls_model.find('sklearn') != -1:
            print('##### Fitting the model #####')
            if hasattr(self.model, 'partial_fit'):
                self.model.partial_fit(x_train, y_train.ravel(), np.unique(y_train))
            else:
                self.model.fit(x_train, y_train.ravel())
        else:
            raise NotImplementedError

    def torch_train(self, model, train_loader, val_loader, **kwds):
        """Default train function for PyTorch models"""
        epochs = kwds['epochs']
        loss_func = nn.BCELoss()
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.model.train()
        for epoch in range(1, epochs+1):
            mean_train_loss = 0.0
            for i, (xs, ys) in enumerate(train_loader):
                optim.zero_grad() # reset gradients
                outputs = model(xs)
                train_loss = loss_func(outputs, ys)
                train_loss.backward() # gradient back propagation
                optim.step()
                mean_train_loss = (mean_train_loss * i + float(train_loss)) / (i + 1)

            mean_val_loss = 0.0
            for i, (xs, ys) in enumerate(val_loader):
                outputs = model(xs)
                val_loss = loss_func(outputs, ys)
                mean_val_loss = (mean_val_loss * i + float(val_loss)) / (i + 1)
            print('Epoch {}\nMean train/validation loss: {:.4f}/{:.4f}'.format(epoch, mean_train_loss, mean_val_loss))

    def perform_metrics(self, x_train, y_train, x_val, y_val, thresh):
        print('\n')
        if thresh == 0:
            thresh = 0.15
        #threshold as argument when calling function
        opt = False
        if self.cls_model.find('torch') != -1:
            #PyTorch
            otp = True
            x_train = torch.tensor(x_train).float()
            y_train = torch.tensor(y_train, dtype=torch.float)
            x_val = torch.tensor(x_val).float()
            y_val = torch.tensor(y_val, dtype=torch.float)

            self.model.eval()

            threshold = torch.Tensor([thresh])

            train_output = self.model(x_train)
            train_target = y_train
            predict_train = (train_output>threshold).float()*1

            val_output = self.model(x_val)
            val_target = y_val
            predict_val = (val_output>threshold).float()*1

            train_output = train_output.detach().numpy()
            train_target = train_target.detach().numpy()
            val_output = val_output.detach().numpy()
            val_target = val_target.detach().numpy()

        elif self.cls_model.find('sklearn') != -1:
            #scikit
            train_target = y_train
            val_target = y_val

            if hasattr(self.model, 'predict_proba'):
                otp = True
                train_output = self.model.predict_proba(x_train)[:,1]
                val_output = self.model.predict_proba(x_val)[:,1]

            predict_train = self.model.predict(x_train)
            predict_val = self.model.predict(x_val)

        elif self.cls_model.find('keras') != -1:
            #Keras
            otp = True
            train_target = y_train
            val_target = y_val

            train_output = self.model.predict(x_train)
            val_output = self.model.predict(x_val)

            predict_train = (train_output > thresh)
            predict_val = (val_output > thresh)

        else:
            raise NotImplementedError


        print("########################")
        print("Metrics on training set")
        print("########################\n")
        # adding this condition since, in scikit, it is not always possible to calculate roc_auc
        if otp:
            auc_train = metrics.roc_auc_score(train_target, train_output)
            print("AUC: {}\n".format(auc_train))

        conf_matrix_train = metrics.confusion_matrix(train_target, predict_train)

        print("Confusion Matrix:\n\n{}\n".format(conf_matrix_train))
        print("Classification Report:\n\n{}".format(metrics.classification_report(train_target, predict_train)))

        print("########################")
        print("Metrics on validation set")
        print("########################\n")
        if otp:
            auc_val = metrics.roc_auc_score(val_target, val_output)
            print("AUC validation: {}\n".format(auc_val))
        conf_matrix_val = metrics.confusion_matrix(val_target, predict_val)
        print("Confusion Matrix:\n\n{}\n".format(conf_matrix_val))
        print("Classification Report:\n\n{}".format(metrics.classification_report(val_target, predict_val)))

    def predict(self, x_test, y_test):
        "Predict API of the trainer"
        if self.cls_model.find('keras') != -1:
            pred = self.model.predict(x_test)
            baseline_results = self.model.evaluate(x_test, y_test, verbose=0)
            print('\n')
            for name, value in zip(self.model.metrics_names, baseline_results):
                print(name, 'test: ', value)
        elif (self.cls_model.find('sklearn') != -1 or self.cls_model.find('xgboost') != -1):
            pred = self.model.predict_proba(x_test)[:,1]
        elif self.cls_model.find('torch') != -1:
            self.model.eval()
            pred = self.model(x_test)
            pred = pred.detach().numpy()
        else:
            raise NotImplementedError
        return pred

    def save(self, fout):
        "Save our model to given file"
        if self.cls_model.find('keras') != -1:
            self.model.save(fout) #format .h5
        elif self.cls_model.find('torch') != -1 and torch:
            torch.save(self.model.state_dict(), fout) #format .pth
        elif self.cls_model.find('xgboost') != -1:
            self.model.save_model(fout) #format .json
        elif self.cls_model.find('sklearn') != -1:
            pickle.dump(self.model, open(fout, 'wb')) #format .pkl
        else:
            raise NotImplementedError

def train_model(model, files, labels, preproc=None, params=None, specs=None, fout=None, dtype=None):
    """
    Train given model on set of files, params, specs

    :param model: the model class
    :param files: the list of files to use for training
    :param labels: the list of label files to use for training or label name to use in data
    :param preproc: file name which contains preprocessing function
    :param params: list of parameters to use for training (via input json file)
    :param specs: file specs
    :param fout: output file name to save the trained model
    """
    met = False
    threshold = 0
    if not params:
        params = {}
    if not specs:
        specs = {}
    model = load_code(model, 'model')

    if len(inspect.getfullargspec(model).args) > 0:
        scikit_xg = False
    else:
        scikit_xg = True

    if preproc:
        preproc = json.load(open(preproc))
    if file_type(files) == 'root':
        gen = RootDataGenerator(files, labels, params, preproc, specs)
    else:
        gen = MetaDataGenerator(files, labels, params, preproc, dtype)

    epochs = None
    torch_ = False

    if 'epochs' in params:
        epochs = params.get('epochs', 10)
    batch_size = params.get('batch_size', 50)
    shuffle = params.get('shuffle', True)

    if 'metrics' in params:
        if isinstance(params['metrics'], dict):
            threshold = params['metrics']['threshold']
            met = True
        else:
            if params['metrics'] == True:
                met = True
            else:
                met = False

    split = params.get('split', 0.3)
    trainer = False

    if epochs:
        kwds = {'epochs': epochs, 'batch_size': batch_size,
                'shuffle': shuffle}
    else:
        kwds = {'batch_size': batch_size,'shuffle': shuffle}


    for data in gen:
        time_ml = time.time()
        if np.shape(data[0])[0] == 0:
            print("received empty x_train chunk")
            break
        if len(data) == 2:
            x_train = data[0]
            y_train = data[1]
        elif len(data) == 3: # ROOT data with mask array
            x_train = data[0]
            x_mask = data[1]
            x_train[np.isnan(x_train)] = 0 # convert all nan's to zero
            y_train = data[2]

        print("x_train chunk of {} shape".format(np.shape(x_train)))
        print("y_train chunk of {} shape".format(np.shape(y_train)))
        if len(data) == 3:
            print("x_mask chunk of {} shape".format(np.shape(x_mask)))
        if not trainer:
            if not scikit_xg:
                idim = np.shape(x_train)[-1] # read number of attributes we have
                model = model(idim) #PyTorch or Keras
                if str(type(model)).lower().find('torch') != -1:
                    torch_ = True
            else:
                model = model()

            trainer = Trainer(model, verbose=params.get('verbose', 0))

        # convert y_train to categorical array
        if hasattr(model, 'categorical_crossentropy'):
            if model.loss == 'categorical_crossentropy':
                y_train = to_categorical(y_train)
        x_train = np.append(x_train, np.array(y_train).reshape(len(y_train), 1), axis=1)

        #create the test set
        train_val, test = train_test_split(x_train, stratify=y_train,test_size=0.2, random_state=21, shuffle=True)
        X_train_val = train_val[:,:-1]
        Y_train_val = train_val[:,-1:]
        X_test = test[:,:-1]
        Y_test = test[:,-1:]

        #create the validation set
        train, val = train_test_split(train_val, stratify=Y_train_val, test_size=0.2, random_state=21, shuffle=True)
        X_train=train[:,:-1]
        Y_train=train[:,-1:]
        X_val=val[:,:-1]
        Y_val=val[:,-1:]

        #fit the model
        #print(f"\n####Time pre ml: {time.time()-time_ml}")
        print('\n')
        time0 = time.time()

        #training the model using train classifier
        if torch_:
            train_tensor = torch.utils.data.TensorDataset(torch.tensor(X_train).float(), torch.tensor(Y_train, dtype=torch.float))
            test_tensor  = torch.utils.data.TensorDataset(torch.tensor(X_test).float(),torch.tensor(Y_test, dtype=torch.float))
            eval_tensor  = torch.utils.data.TensorDataset(torch.tensor(X_val).float(),torch.tensor(Y_val, dtype=torch.float))
            train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(eval_tensor, batch_size=batch_size, shuffle=False)
            test_loader = torch.utils.data.DataLoader(test_tensor, batch_size=batch_size, shuffle=False)

            #trainer.fit(train_loader, test_loader)
            trainer.fit(train_loader, val_loader, **kwds)

        else:
            trainer.fit(X_train, Y_train, **kwds, validation_data=(X_val,Y_val))

        if met:
            trainer.perform_metrics(X_train, Y_train, X_val, Y_val, threshold)

        d = trainer.predict(X_test, Y_test)
        fpr, tpr, thresholds = metrics.roc_curve(Y_test, d)
        # calculate the g-mean for each threshold
        gmeans = np.sqrt(tpr * (1-fpr))
        # locate the index of the largest g-mean
        ix = np.argmax(gmeans)
        print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

    print(f"\n####Time for training: {time.time()-time0}\n")

    if fout and hasattr(trainer, 'save'):
        trainer.save(fout)
