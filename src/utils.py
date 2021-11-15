import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd

from sklearn.preprocessing import StandardScaler

torch.manual_seed(27)

def get_data():
    '''
    Read and preprocess the dataset.

    Returns:
    3-tuple containing 3 2-tuples: (train_x, train_y), (val_x, val_y), (test_x, 
    test_y), with a 70:15:15 ratio.
    '''
    # TODO: We can probably do it cheaper without dataframes? This is currently 
    # copied straight out of Matheus's Collab notebook.
    # TODO: Parametrize file path
    app_df = pd.read_csv('../data/application_data.csv')
    # One hot encode
    to_encode = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
               'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 
               'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE',
               'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE' ]

    encoded = pd.get_dummies(to_encode) # this is not correct, check
    app_df.drop(to_encode, axis = 1, inplace  = True)
    app_df.merge(encoded, left_index= True, right_index = True)

    # Fill NaN

    app_df = app_df.fillna(app_df.mean())

    # Separate X and Y into arrays 

    X = app_df.drop('TARGET', axis = 1).values.astype('float32')
    y = app_df.TARGET.values.astype('float32')

    # Normalize X
    sc = StandardScaler()
    X = sc.fit_transform(X)

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    # Train-val-test split
    num_examples = X.size()[0]
    shuffle_ix = torch.randperm(num_examples)
    first_valid_ix = int(num_examples * 0.7)
    last_valid_ix = int(num_examples * 0.85)
    train_X = X[shuffle_ix][:first_valid_ix]
    train_y = y[shuffle_ix][:first_valid_ix]

    valid_X = X[shuffle_ix][first_valid_ix:last_valid_ix]
    valid_y = y[shuffle_ix][first_valid_ix:last_valid_ix]

    test_X = X[shuffle_ix][last_valid_ix:]
    test_y = y[shuffle_ix][last_valid_ix:]
    return ((train_X, train_y), (valid_X, valid_y), (test_X, test_y))

def get_fc_nn(layer_sizes, activation):
    '''
    Instantiate a fully connected feedforward neural network.

    Parameters:

        *layer_sizes is an array of positive integers representing the number of 
    units in each layer. Must be of length at least two, since layer_sizes[0] 
    is taken to be the number of input variables and layer_sizes[-1] is the 
    number of output variables.
        *activation is an activation (e.g. nn.ReLU) to be applied to each 
    hidden layer of the model.
        *loss_function is a function to be applied to the last layer, and the 
    corresponding value(s) of train_y in order to calculate the loss of the 
    model.

    Returns:
    A 2-tuple containing a new instance of a fully connected feedforward neural 
    network.
    '''
    layers = []
    num_layers = len(layers)
    for ix, layer_size in enumerate(layer_sizes[1:]):
        layers.append(nn.Linear(layer_sizes[ix], layer_size))
        # Only add activation if not last layer. Otherwise, use sigmoid
        if ix < num_layers - 1:
            layers.append(activation())
        else:
            layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)

def train_nn(model, train_x, train_y, valid_x, valid_y, batch_size, criterion, 
             lr, momentum, num_epochs):
    '''
    Train a neural network using Stochastic Gradient Descent.

    Parameters:

        *model is an instance of an nn.Module subclass.
        *train_x is a tensor representing the training data's attributes.
        *train_y is a tensor representing the training data's labels.
        *valid_x is a tensor representing the validation data's attributes.
        *valid_y is a tensor representing the validation data's labels.
        *batch_size is the size of the minibatch.
        *criterion is a function to be applied to the model output and the 
    train labels in order to calculate the loss of the model.
        *lr is a float representing the learning rate.
        *lr is a float representing the training momentum.
        *num_epochs is an integer representing the number of epochs.

    Returns:
    A 2-tuple containing a trained instance of an nn.Module subclass, and the 
    training history
    '''
    shuffle_ix = torch.randperm(train_x.size()[0])
    train_x = train_x[shuffle_ix]
    train_y = train_y[shuffle_ix]
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    len_train_x = len(train_x)
    for epoch in range(num_epochs):
        running_loss = 0.0
        i = 0
        batch_start = i * batch_size
        while batch_start < len_train_x:
            batch_end = min(batch_start + batch_size, len_train_x)
            inputs = train_x[batch_start:batch_end]
            labels = train_y[batch_start:batch_end].unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss))
                running_loss = 0.0
            if i == 0 and epoch == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss))
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            i += 1
            batch_start += batch_size
    torch.save(model.state_dict(), 'saved_models/mini.torch')
    return model

def load_nn(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
