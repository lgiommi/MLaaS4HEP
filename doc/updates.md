## New features of MLaaS4HEP

The MLaaS4HEP code has been updated to support new features making it more flexible for the user to train new Machine Learning (ML) and Deep Learning (DL) models.
Specifically, these new updates have made it possible to:

- generalize MLaaS4HEP to other ML frameworks,
- compute ML performance metrics,
- improve the MLaaS4HEP training method.

### Generalization to other ML frameworks and libraries

MLaaS4HEP has already been tested using a Sequential NN  written in Keras. However, among its main objectives, MLaaS4HEP aims to be ML  framework agnostic, which means the user can decide to use any ML framework and algorithm, and, in order to get closer to this goal, some changes have been made in order to support other frameworks and libraries.
In particular, PyTorch, Scikit-Learn and XGBoost were correctly integrated within the MLaaS4HEP code.

In the following we will show how to define ML/DL models using the frameworks mentioned above.

The ML/DL model definition must be done in an external python file.
Firstly, within the python file the user needs to import the libraries needed to define
the model. Once this is done, the user moves on to the model definition.
However, it is necessary for the model definition to occur within a function
called `model` which, depending on the kind of model, may take an argument.

Indeed, if the defined model is a Neural Network, the function should have `idim`
as argument, which is used to specify the input shape required to define
the Neural Network. However, if the model is not a Neural Network,
the function `model` has no arguments.
Finally, once the model is defined, the function needs to return the classifier.

To clarify what has been described so far, four examples of model definitions are given below.

#### 1. Sequential Neural Network defined with Keras

```
from keras.models import Sequential
from keras.layers import Dense, Activation

def model(idim):
    ml_model = Sequential([
        Dense(32, input_shape=(idim,)),
        Activation('relu'),
        Dense(2), # use Dense(1) if you have 2 output classes
        Activation('softmax'),
    ])
    ml_model.compile(optimizer='adam', \
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return ml_model
```

#### 2. Sequential Neural Network defined with PyTorch

The most common way
to use a PyTorch model is to provide a class definition of it, containing the structure
of the model and how it must be trained, and this can be done easily by using PyTorch
nn.Module class.
In this file, the user can define a custom method for training the model: in this case, the training method should be called `torch_train. If no
training method is provided, a default one is selected.

```
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
        self.inference_mode = True
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

    def torch_train(self, model, train_loader, val_loader):
        epochs = 5
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

def model(idim):
    torch_model = ClassifierNN(idim)
    return torch_model
```

#### 3. Gradient Boosting Classifier defined with Scikit-Learn

```
from sklearn.ensemble import GradientBoostingClassifier

def model():
    gbc = GradientBoostingClassifier(n_estimators=10, max_depth=3,min_samples_leaf=50)
    return gbc
```

#### 4. Defining an XGBoost Classifier

```
import xgboost as xgb

def model():
    params = {
        'objective':'binary:logistic',
        'max_depth': 5,
        'alpha': 10,
        'learning_rate': 0.001,
        'n_estimators': 200
    }
    xgb_model = xgb.XGBClassifier(**params)
    return xgb_model
```

#### Setting `params.json`


The `params.json` file stores all the parameters on which MLaaS4HEP relies, such as chunk size, batch size, number of epochs and so on, and for this reason it is important to set it in the appropriate way depending on the model used.
An example of `params.json` is shown below.

```
{
    "nevts": 100000,
    "chunk_size": 25000,
    "epochs": 3,
    "shuffle": true,
    "batch_size": 100,
    "identifier": "",
    "branch": "myTree",
    "selected_branches":"",
    "exclude_branches":"",
    "hist": "pdfs",
    "redirector": "",
    "verbose": 1
  }
```

However, the given example shows some parameters that need to be changed
depending on the model used: `epochs` and `chunk_size`.
The presence of these parameters assumes that the defined model is capable of incremental learning and that it `knows` the concept of epoch.
A `params.json` defined in this way is therefore suitable if the model under consideration is a Neural Network.
However, if the defined model does not satisfy the two previous
conditions
it is necessary to remove the `epochs` key and change the value
associated with the `chunk_size` key by setting it to -1:
in this way, the training of the model will be carried out using
a single chunk containing all events.

Thus, the previous example is suitable for examples 1. and 2.,
whereas if a model is defined as in examples 3. and 4., the params.json file will need to be modified as follows.

```
{
    "nevts": 100000,
    "chunk_size": -1,
    "shuffle": true,
    "batch_size": 100,
    ...
  }
```

#### Other examples

###### 1. Scikit-learn MLPClassifier

```
from sklearn.neural_network import MLPClassifier

def model():
    mlp = MLPClassifier(solver="adam", learning_rate="adaptive", max_iter=350)
    return mlp
```

###### 2. Scikit-learn AdaBoost

```
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def model():
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100, algorithm="SAMME.R", learning_rate=0.5)
    return ada
```

###### 3. Scikit-learn RandomForest
```
from sklearn.ensemble import RandomForestClassifier

def model():
    rf = RandomForestRegressor(n_estimators=1000, random_state=42)
    return rf
```

###### 4. Scikit-learn DecisionTree

```
from sklearn.tree import DecisionTreeClassifier

def model():
    dt = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=42)
    return dt
```

###### 5. Scikit-learn KNN
```
from sklearn.neighbors import KNeighborsClassifier

def model():
    knn = KNeighborsClassifier(n_neighbors=20)
    return knn
```

###### 6. Scikit-learn Linear SVM
```
from sklearn import svm

def model():
    svc = svm.SVC(kernel="linear", C=0.025)
    return svc
```

###### 7. Scikit-learn Gaussian Process Classifier

```
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

def model():
   gpc = GaussianProcessClassifier(kernel=1*RBF(1.0))
   return gpc
```

###### 8. Scikit-learn Logistic Regression

```
from sklearn.linear_model import LogisticRegression

def model():
   lr = LogisticRegression(penalty='l1', random_state=42)
   return lr
```

### Supporting performance Metrics

Since metrics are used to quantify the learning abilities of the models, it may be useful
using MLaaS4HEP to have a general overview of the most common metrics. To provide
such information, the new method called `performance_metric has been defined in the
code, regardless of the used ML framework. In particular, it was decided to provide the
following metrics:

- AUC,
- Confusion Matrix,
- Classification Report.

The performance metrics function prints the values related to the previous metrics
both on the training and on the validation set for each chunk and has been implemented
in such a way that it is the user who decides whether to execute this function or not: if
the user is interested in obtaining the scores of these metrics, it is necessary to specify this
request in the `params.json file`, by inserting the key `metrics`.

Moreover, an additional
parameter, was included within this key to obtain these scores:
the threshold.
In case the threshold is not defined,
it will be set to the default value (0.5).

In the below example, two cases are provided: one in which the threshold
is not defined and one in which it is defined instead.

```
{
    ...
    "metrics": true,
    or
    "metrics": {"threshold": 0.5},
    ...
  }
```

### Improvements on the training method

The current MLaaS4HEP training procedure, when a NN model is chosen, is performed  chunk by chunk where each chunk is used to train the model for *n* epochs, with *n* defined  by the user.
It has been introduced an additional training procedure, the standard one,  where each epoch is performed using all the chunks.  Then the training continues for *n* times.

Now it is possible to select the training method you want to apply, and to do so you need to insert a new key within the params.json file.
The new key to be entered is called `training` and depending on the value associated with it, training will be carried out with either the original or the standard method.
In case the user decides to apply the newly introduced method, he/she needs to associate the new key with the value `standard`.
If the value associated with the key is different or the key is not present, the original training mode will be carried out.

Below is an example of the params.json file also containing this new key.

```
{
    "nevts": 100000,
    "shuffle": true,
    "chunk_size": 25000,
    "epochs": 3,
    "batch_size": 100,
    "training": "standard",
    ...
  }
```
