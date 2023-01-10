import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments
    parser.add_argument('--num_of_comm', type=int, default=10, help='rounds of training')
    parser.add_argument('--num_of_clients', type=int, default=10, help='the number of clients')
    parser.add_argument('--frac', type=int, default=0.3, help='the fraction of clients')
    parser.add_argument('--local_epoch', type=int, default=5, help='the number of local epochs')
    parser.add_argument('--local_batchsize', type=int, default=100, help='local batch size')
    parser.add_argument('--batch_size', type=int, default=100, help='test batch size')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='learning rate')

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset name')
    parser.add_argument('--iid', action='store_true', default=True, help='iid or non-iid')
    parser.add_argument('--gpu', type=str, default='0', help='gpu id, -1 for cpu')

    # dp arguments
    parser.add_argument('--noise', type=int, default=0, help='0-no noise; 1-laplace, 2-gaussian')
    parser.add_argument('--sigma', type=float, default=0.02, help='sigma of noise')

    args = parser.parse_args()

    return args
