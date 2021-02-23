# from options.train_options import TrainOptions
from model import Model
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', type=str, default='../data/')

    parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
    parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
    parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

    # for training
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--total_iter', type=int, default=100, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
    parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
    parser.add_argument('--batch_size', type=int, default=2, help='# of batch_szie')
    parser.add_argument('--workers', type=int, default=1, help='# of workers')
    parser.add_argument('--niter_decay', type=int, default=50, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

    # for text_encoder
    parser.add_argument('--img_size', type=int, default=256, help='img_size')
    parser.add_argument('--vocab_size', type=int, default=300, help='# of vobabularies')
    parser.add_argument('--words_num', type=int, default=18, help='# of vobabularies')
    parser.add_argument('--captions_num', type=int, default=5, help='# of vobabularies')
    
    parser.add_argument('--embedding_dim', type=int, default=300, help='word embedding dimension')
    parser.add_argument('--w_dim', type=int, default=1024, help='word feature encoding dimension')
    parser.add_argument('--num_layers', type=int, default=1, help='# of layers of RNN text encoder')

    # for image encoder

    # for generator
    parser.add_argument('--channels', type=int, default=1024, help='# of channels of img feature')
    parser.add_argument('--q_dim', type=int, default=512, help='question embedding dimension')

    # for discriminators
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')

    args = parser.parse_args()
    model = Model(args)
    model.train()
    # import numpy as np

    # data_iter = iter(model.dataloader)
    # data = data_iter.next()
    # print(np.shape(data[0]))
    