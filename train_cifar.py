import glob, os, argparse, subprocess
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.debugging.set_log_device_placement(False)

parser = argparse.ArgumentParser(description='')
parser.add_argument("--train_path", default="../test", type=str, help = 'abc')
parser.add_argument("--data_path", type=str, help = 'home directory of ILSVRC2012')
parser.add_argument("--arch", default='ResNet-56', type=str)
parser.add_argument("--dataset", default='CIFAR10', type=str)

parser.add_argument("--learning_rate", default = 1e-2, type=float)
parser.add_argument("--decay_points", default = [.3, .6, .8], type=float, nargs = '+')
parser.add_argument("--decay_rate", default=.2, type=float)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--batch_size", default = 128, type=int)
parser.add_argument("--val_batch_size", default=256, type=int)
parser.add_argument("--train_epoch", default=100, type=int)

parser.add_argument("--pruning", default=True, action = 'store_true')
parser.add_argument("--trained_param", default='pretrained/res56_cifar10.pkl', type=str)
parser.add_argument("--target_rate", default=.45, type=float)
parser.add_argument("--search_step", default=.2, type=float)
parser.add_argument("--minimum_rate", default=.0, type=float)
parser.add_argument("--num_teacher", default=5, type=int)
parser.add_argument("--Knowledge", default=False, action = 'store_true')

parser.add_argument("--gpu_id", default= [0], type=int, nargs = '+')
parser.add_argument("--do_log", default=200, type=int)
parser.add_argument("--compile", default=True, action = 'store_true')
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

def get_avg_plot(base_path):
    pathes = glob.glob(os.path.join(base_path, '*'))
    summary_writer = tf.summary.create_file_writer(os.path.join(base_path, 'average'))
    
    train_acc = []
    test_acc = []
    for path in pathes:
        logs = np.load(os.path.join(path,'logs.npy'), allow_pickle=True)[()]
        train_acc.append(logs['train'])
        test_acc.append(logs['test'])
    train_acc = np.mean(np.vstack(train_acc),0)
    test_acc = np.mean(np.vstack(test_acc),0)

    with summary_writer.as_default():    
        for i, (train, test) in enumerate(zip(train_acc, test_acc)):
            tf.summary.scalar('Accuracy/train', train, step=i+1)
            tf.summary.scalar('Accuracy/test', test, step=i+1)

    with open(os.path.join(base_path, 'average', 'acc.txt'),'w') as f:
        f.write('train_acc: %f\n'%train_acc[-1])
        f.write('test_acc: %f'%test_acc[-1])

if __name__ == '__main__':
    subprocess.call('python pruning.py' + args_to_text(args), shell=True)

    args.trained_param = os.path.join(args.train_path, 'slimmed_params.pkl')
    args.pruning = False
    args.Knowledge = True
    subprocess.call('python train.py' + args_to_text(args), shell=True)
