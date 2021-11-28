import utils

import torch
import torch.nn as nn

TRAIN_PATH = '../data/application_data_train.csv'
VALID_PATH = '../data/application_data_valid.csv'
TEST_PATH = '../data/application_data_test.csv'
MINI_PATH = '../data/application_data_mini.csv'

NN_LAYERS = [205, 900, 1]
NONLINEARITY = nn.Sigmoid
NN_NAME = 'mini_test'

def curriculum_train(model, train_x, train_y, valid_x, valid_y, batch_size,
                     criterion, lr, momentum, num_epochs):
    '''
    Repeatedly train model with different training sets
    In this first version, we first overfit the model to a small balanced 50:50
    pos:neg ratio, then the true ratio (about 92:8)
    '''
    small_neg_train_x = train_x[train_y == 0][-300:]
    small_pos_train_x = train_x[train_y == 1][-300:]
    small_neg_train_y = train_y[train_y == 0][-300:]
    small_pos_train_y = train_y[train_y == 1][-300:]
    first_train_x = torch.cat((small_neg_train_x, small_pos_train_x), axis=0)
    first_train_y = torch.cat((small_neg_train_y, small_pos_train_y), axis=0)
    shuffle_ix = torch.randperm(600)
    first_train_x = first_train_x[shuffle_ix]
    first_train_y = first_train_y[shuffle_ix]
    utils.train_nn(model, first_train_x, first_train_y, valid_x, valid_y,
                   batch_size, criterion, lr, momentum, num_epochs // 3)
    utils.eval_nn(model, first_train_x, first_train_y, valid_x, valid_y, criterion)

    second_train_x = train_x[:600]
    second_train_y = train_y[:600]
    utils.train_nn(model, second_train_x, second_train_y, valid_x, valid_y,
                   batch_size, criterion, lr, momentum, 2 * num_epochs // 3)
    utils.eval_nn(model, second_train_x, second_train_y, valid_x, valid_y, criterion)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    (train_X, train_y) = utils.get_data_alt(MINI_PATH, remove_outliers=True, remove_correlated_columns=True, device=device)
    (valid_X, valid_y) = utils.get_data_alt(VALID_PATH, remove_correlated_columns=True, device=device)
    print('TRAIN SHAPES')
    print(train_X.shape)
    print(train_y.shape)
    print('VALIDATION SHAPES')
    print(valid_X.shape)
    print(valid_y.shape)
    print(f'Training network with {NN_LAYERS} layers, {NONLINEARITY} nonlinearity, {NN_NAME} name.')

    model = utils.get_fc_nn(NN_LAYERS, NONLINEARITY, device)

    batch_size = 1
    criterion = nn.BCELoss()
    lr = 0.001
    momentum = 0.9
    num_epochs = 50
    model, training_history = utils.train_nn(model, train_X, train_y, valid_X, valid_y,
            batch_size, criterion, lr, momentum, num_epochs, NN_NAME, device)

if __name__ == '__main__':
    main()
