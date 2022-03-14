import os, argparse, subprocess
import glob
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser(description='')
parser.add_argument("--train_path", default="../test", type=str, help = 'abc')
parser.add_argument("--data_path", type=str)
parser.add_argument("--arch", default='ResNet-56', type=str)
parser.add_argument("--dataset", default='CIFAR10', type=str)
parser.add_argument("--trained_param", default = 'pretrained/res56_cifar10.pkl', type=str)

parser.add_argument("--Knowledge", default=False, action = 'store_true')
parser.add_argument("--num_teacher", default=5, type=int)

parser.add_argument("--learning_rate", default = 1e-2, type=float)
parser.add_argument("--decay_points", default = [.3, .6, .8], type=float, nargs = '+')
parser.add_argument("--decay_rate", default=.2, type=float)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--batch_size", default = 128, type=int)
parser.add_argument("--val_batch_size", default=256, type=int)
parser.add_argument("--train_epoch", default=100, type=int)

parser.add_argument("--searching", default=True, action = 'store_true')
parser.add_argument("--search_target_rate", default=.45, type=float)
parser.add_argument("--search_step", default=.2, type=float)
parser.add_argument("--search_batch_size", default=256, type=int)
parser.add_argument("--search_seed", default=1, type=int)

parser.add_argument("--gpu_id", default= [0], type=int, nargs = '+')
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--accum", default=1, type=int)
parser.add_argument("--do_log", default=200, type=int)
parser.add_argument("--compile", default=False, action = 'store_true')
parser.add_argument("--setting", default=None, type=str)
args = parser.parse_args()

def args_to_text(args):
    targs = ''
    args = args.__dict__
    for k in args:
        if isinstance(args[k], list):
            args[k] = "".join(map(lambda x: str(x) + ' ', args[k]))

        if isinstance(args[k], bool):
            if args[k]:
                args[k] = ''
            else:
                continue

        targs += ' --%s %s'%(k, args[k])
    return targs

if __name__ == '__main__':
    subprocess.call('python pruning.py' + args_to_text(args), shell=True)        
    for k in list(args.__dict__.keys()):
        if 'search' in k:
            delattr(args, k)
    args.trained_param = os.path.join(args.train_path, 'slimmed_params.pkl')
    subprocess.call('python train.py' + args_to_text(args), shell=True)
