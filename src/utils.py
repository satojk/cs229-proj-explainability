import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

ALL_DATA_PATH = '../data/application_data.csv'

torch.manual_seed(0)

def train_valid_test_split():
    np.random.seed(0)
    df = pd.read_csv(ALL_DATA_PATH)
    # One hot encode
    to_encode = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
               'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
               'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE',
               'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE' ]

    encoded = pd.get_dummies(df[to_encode])
    df = df.drop(to_encode, axis=1)
    df = df.merge(encoded, left_index=True, right_index=True)

    df = df.sample(frac=1, random_state=0, ignore_index=True)
    train, valid, test = np.split(df,
               [int(.7*len(df)), int(.85*len(df))])
    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    test = test.reset_index(drop=True)
    mini = train.sample(frac=0.005, random_state=0, ignore_index=True)

    train.to_csv('../data/application_data_train.csv', index=False)
    valid.to_csv('../data/application_data_valid.csv', index=False)
    test.to_csv('../data/application_data_test.csv', index=False)
    mini.to_csv('../data/application_data_mini.csv', index=False)

def get_data_alt(path):
    '''
    Load and preprocess the dataset.

    Arguments:
    -path: the file path for the dataset to be loaded and preprocessed.

    Returns:
    2-tuple containing the X values and the y values of the dataset in the
    given path
    '''
    # TODO: We can probably do it cheaper without dataframes? This is currently
    # copied straight out of Matheus's Collab notebook.
    app_df = pd.read_csv(path)

    # Fill NaN

    app_df = app_df.fillna(app_df.mean())

    # Drop ID

    app_df = app_df.drop('SK_ID_CURR', axis = 1)

    X = app_df.drop('TARGET', axis = 1).values.astype('float32')
    y = app_df.TARGET.values.astype('float32')

    sc = StandardScaler()
    X = sc.fit_transform(X)

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    return X, y

def get_data(path):
    '''
    Load and preprocess the dataset.

    Arguments:
    -path: the file path for the dataset to be loaded and preprocessed.

    Returns:
    2-tuple containing the X values and the y values of the dataset in the
    given path
    '''
    # TODO: We can probably do it cheaper without dataframes? This is currently
    # copied straight out of Matheus's Collab notebook.
    app_df = pd.read_csv(path)

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

    return X, y

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
    A new instance of a fully connected feedforward neural network.
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
             lr, momentum, num_epochs, model_name):
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
        *model_name is a string. The resulting model will be saved as
        f'saved_models/{model_name}.torch'

    Returns:
    A 2-tuple containing a trained instance of an nn.Module subclass, and the
    training history
    '''
    shuffle_ix = torch.randperm(train_x.size()[0])
    train_x = train_x[shuffle_ix]
    train_y = train_y[shuffle_ix]
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    len_train_x = len(train_x)
    training_history = {
        'epoch': [],
        'train_acc': [],
        'train_loss': [],
        'valid_acc': [],
        'valid_loss': [],
    }
    for epoch in range(num_epochs):
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
            i += 1
            batch_start += batch_size

        # print statistics
        train_loss, train_acc, valid_loss, valid_acc = eval_nn(model,
                train_x, train_y, valid_x, valid_y, criterion)
        print('[epoch %d]\ttrain L: %.3f\tvalid L: %.3f\ttrain A: %.3f\tvalid A: %.3f' %
              (epoch + 1, train_loss, valid_loss, train_acc, valid_acc))
        training_history['epoch'].append(epoch)
        training_history['train_acc'].append(train_acc.item())
        training_history['train_loss'].append(train_loss.item())
        training_history['valid_loss'].append(valid_loss.item())
        training_history['valid_acc'].append(valid_acc.item())

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'saved_models/{model_name}_{epoch}.torch')
            with open(f'saved_models/{model_name}_training_history.json', 'w') as f:
                f.write(json.dumps(training_history))
    torch.save(model.state_dict(), f'saved_models/{model_name}_{num_epochs}.torch')
    with open(f'saved_models/{model_name}_training_history.json', 'w') as f:
        f.write(json.dumps(training_history))
    return model, training_history

def eval_nn(model, train_x, train_y, test_x, test_y, criterion):
    '''
    Evaluate a given model according to accuracy and the training criterion,
    for both train set and a test set.
    '''
    train_loss = criterion(model(train_x), train_y.unsqueeze(1))
    train_acc = sum(torch.round(model(train_x)) == train_y.unsqueeze(1)) / train_x.shape[0]
    test_loss = criterion(model(test_x), test_y.unsqueeze(1))
    test_acc = sum(torch.round(model(test_x)) == test_y.unsqueeze(1)) / test_x.shape[0]
    return (train_loss, train_acc, test_loss, test_acc)

def load_nn(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model