from options.train_options import TrainOptions
from model import Model


if __name__ == '__main__':
    global TrainOptions
    train_options = TrainOptions()
    opt = train_options.parse(save=True)
    model = Model(opt)