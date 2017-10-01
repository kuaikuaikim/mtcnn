import argparse
import sys
sys.path.append('..')
from core.imagedb import imagedb
from train import train_p_net
from core.model import P_Net

data_name='wider'
model_path='../data/%s_model/pnet'%data_name

def train_P_net(image_set, root_path, dataset_path, train_images_root_path,train_annotation_file_path, prefix,
                end_epoch, frequent, lr, batch_size):

    imdb = imagedb(train_images_root_path, train_annotation_file_path)
    gt_imdb = imdb.gt_imdb()
    gt_imdb = imdb.append_flipped_images(gt_imdb)

    train_p_net(prefix, end_epoch, gt_imdb,batch_size,frequent, lr)

def parse_args():
    parser = argparse.ArgumentParser(description='Train proposal net(12-net)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--image_set', dest='image_set', help='training set',
    #                     default='train_12', type=str)
    # parser.add_argument('--root_path', dest='root_path', help='output data folder',
    #                     default='../data', type=str)
    # parser.add_argument('--dataset_path', dest='dataset_path', help='dataset folder',
    #                     default='../data/%s'%data_name, type=str)
    parser.add_argument('--model_path', dest='prefix', help='new model prefix',
                        default='./model', type=str)
    parser.add_argument('--train_images_root_path', dest='train_images_root_path', help='output data folder',
                        default='../data', type=str)
    parser.add_argument('--train_annotation_file_path', dest='train_annotation_file_path', help='output data folder',
                        default='../data', type=str)
    parser.add_argument('--end_epoch', dest='end_epoch', help='end epoch of training',
                        default=16, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=200, type=int)
    parser.add_argument('--lr', dest='lr', help='learning rate',
                        default=0.01, type=float)
    parser.add_argument('--batch_size', dest='batch_size', help='train batch size',
                        default=128, type=int)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print 'Called with argument:'
    print args
    train_P_net(args.image_set, args.root_path, args.dataset_path, args.train_images_root_path,args.train_annotation_file_path, args.prefix,
                args.end_epoch, args.frequent, args.lr, args.batch_size)
