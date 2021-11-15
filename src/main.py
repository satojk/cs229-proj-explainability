import utils

import torch.nn as nn

def test_main():
    model = utils.get_fc_nn([105, 600, 600, 1], nn.ReLU)
    model = utils.load_nn(model, 'saved_models/test.torch')
    return model


def main():
    (train, valid, _) = utils.get_data()
    (train_X, train_y) = train
    (valid_X, valid_y) = valid

    model = utils.get_fc_nn([105, 900, 900, 900, 900, 1], nn.Sigmoid)

    batch_size = 1
    criterion = nn.BCELoss()
    lr = 0.001
    momentum = 0.9
    num_epochs = 100
    utils.train_nn(model, train_X[:100], train_y[:100], valid_X, valid_y, batch_size, criterion, lr, momentum, num_epochs)

if __name__ == '__main__':
    main()
