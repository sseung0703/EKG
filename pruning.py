import os, time, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.debugging.set_log_device_placement(False)
from tqdm import tqdm

from dataloader import CIFAR
import op_utils, prune_utils, utils

parser = argparse.ArgumentParser(description='')

parser.add_argument("--train_path", default="test", type=str, help = 'abc')
parser.add_argument("--data_path", default="E:/Datasets/ILSVRC2012", type=str)
parser.add_argument("--arch", default='ResNet-18', type=str)
parser.add_argument("--dataset", default='ILSVRC', type=str)

parser.add_argument("--learning_rate", default = 1e-2, type=float)
parser.add_argument("--decay_points", default = [.3, .6, .8], type=float, nargs = '+')
parser.add_argument("--decay_rate", default=.2, type=float)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--batch_size", default = 128, type=int)
parser.add_argument("--val_batch_size", default=128, type=int)
parser.add_argument("--train_epoch", default=50, type=int)

parser.add_argument("--pruning", default=False, action = 'store_true')
parser.add_argument("--trained_param", type=str)
parser.add_argument("--target_rate", default=.6, type=float)
parser.add_argument("--search_step", default=.2, type=float)
parser.add_argument("--minimum_rate", default=.0, type=float)
parser.add_argument("--num_teacher", default=5, type=int)
parser.add_argument("--Knowledge", default=False, action = 'store_true')

parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--gpu_id", default= [0], type=int, nargs = '+')
parser.add_argument("--accum", default= 1, type=int)
parser.add_argument("--do_log", default=200, type=int)
parser.add_argument("--compile", default=False, action = 'store_true')
parser.add_argument("--setting", default=None, type=str)
args = parser.parse_args()

args.home_path = os.path.dirname(os.path.abspath(__file__))
args.decay_points = [int(dp*args.train_epoch) if dp < 1 else int(dp) for dp in args.decay_points]

if args.dataset == 'ILSVRC':
    args.weight_decay /= len(args.gpu_id)
    args.learning_rate *= args.batch_size/256
    args.val_batch_size = 64 * len(args.gpu_id)

if __name__ == '__main__':
    utils.save_code_and_augments(args)

    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices([tf.config.list_physical_devices('GPU')[i] for i in args.gpu_id], 'GPU')
    for gpu_id in args.gpu_id:
        tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
    devices = ['/gpu:{}'.format(i) for i in args.gpu_id]
 
    strategy = tf.distribute.MirroredStrategy(devices, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    with strategy.scope():
        datasets = CIFAR.build_dataset_providers(args, strategy)
        model = utils.load_model(args, args.num_classes, args.trained_param)

        ## Sub-network search phase
        print ('\nSub-network search phase starts')
        tic = time.time()        
        train_step, train_loss, train_accuracy, optimizer = op_utils.Optimizer(args, model, strategy )
        print ('\nOne epoch fine-tuning')
        for step, data in tqdm(enumerate(datasets['train_sub']), total = args.iter_len['train_sub']):
            optimizer.learning_rate = 1e-3
            train_step(*data)

        GSearch = prune_utils.Greedly_search_get_teacher(args, model, strategy, datasets)
        History = GSearch.run()
        pruned = History.pop()

        searching_time = time.time() - tic
        args.searching_time = searching_time
        print ('Searching time: %f'%searching_time)

        ## Memory bank building phase
        print ('\nMemory bank building phase starts')
        tic = time.time()

        prune_utils.build_memory_bank(args, model, strategy, History, pruned)

        memory_bank_building_time = time.time() - tic
        args.memory_bank_building_time = memory_bank_building_time
        print ('Memory bank building time: %.2f'%memory_bank_building_time)

        ## Save pruned network 
        p,f = prune_utils.set_width(args, model, pruned[0], actual = True)
        utils.save_model(args, model, 'slimmed_params')

        print ([model.Layers[k].kernel.shape[-1] for k in model.Layers if 'conv' in k])
        print ('Pruned params: %.4fM, Pruned FLOPS: %.4fM'%(p/1e6, f/1e6))
        utils.save_code_and_augments(args)
