import utils

import torch
import torch.nn as nn

TRAIN_PATH = '../data/application_data_train.csv'
VALID_PATH = '../data/application_data_valid.csv'
TEST_PATH = '../data/application_data_test.csv'
MINI_PATH = '../data/application_data_mini.csv'
SAVED_PATH = 'saved_models_remote'

NN_LAYERS = [205, 600, 600, 600, 1]
NONLINEARITY = nn.Sigmoid
DROPOUT = 0.5
NN_NAME = '600x3_dropout'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')

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

def train_model():
    print('Using device:', DEVICE)
    (train_X, train_y) = utils.get_data(MINI_PATH, remove_outliers=True, remove_correlated_columns=True, oversample=True, device=DEVICE)
    (valid_X, valid_y) = utils.get_data(VALID_PATH, remove_correlated_columns=True, device=DEVICE)
    print('TRAIN SHAPES')
    print(train_X.shape)
    print(train_y.shape)
    print('VALIDATION SHAPES')
    print(valid_X.shape)
    print(valid_y.shape)
    print(f'Training network with {NN_LAYERS} layers, {NONLINEARITY} nonlinearity, {DROPOUT} dropout, {NN_NAME} name.')

    model = utils.get_fc_nn(NN_LAYERS, NONLINEARITY, DEVICE, dropout=DROPOUT)

    batch_size = 1
    criterion = nn.BCELoss()
    lr = 0.001
    momentum = 0.9
    num_epochs = 30
    model, training_history = utils.train_nn(model, train_X, train_y, valid_X, valid_y,
            batch_size, criterion, lr, momentum, num_epochs, NN_NAME, DEVICE)

def evaluate_model():
    training_history = utils.load_training_history(f'{SAVED_PATH}/{NN_NAME}_training_history.json')
    # Pick best valid AUC
    best_epoch = 0
    best_auc = float('-inf')
    for epoch, auc in enumerate(training_history['valid_auc']):
        if (epoch+1) % 10 == 0 and auc > best_auc:
            best_auc = auc
            best_epoch = epoch
    best_model = utils.get_fc_nn(NN_LAYERS, NONLINEARITY, DEVICE, dropout=DROPOUT)
    best_model = utils.load_nn(best_model, f'{SAVED_PATH}/{NN_NAME}_{best_epoch + 1}.torch')
    (train_X, train_y) = utils.get_data(TRAIN_PATH, remove_outliers=True, remove_correlated_columns=True, oversample=False, device=DEVICE)
    (valid_X, valid_y) = utils.get_data(VALID_PATH, remove_correlated_columns=True, device=DEVICE)
    criterion = nn.BCELoss()
    print(utils.eval_nn(best_model, train_X, train_y, valid_X, valid_y, criterion, DEVICE))
    utils.plot_integrated_gradients(best_model, MINI_PATH, device=DEVICE)

def main():
    # train_model()
    evaluate_model()

if __name__ == '__main__':
    main()
