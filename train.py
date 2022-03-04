import os, time, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
tf.debugging.set_log_device_placement(False)

from dataloader import CIFAR
import op_utils, utils

parser = argparse.ArgumentParser(description='')

parser.add_argument("--train_path", default="test", type=str, help = 'abc')
parser.add_argument("--data_path", type=str, help = 'home directory of ILSVRC2012')
parser.add_argument("--arch", default='ResNet-56', type=str)
parser.add_argument("--dataset", default='CIFAR10', type=str)

parser.add_argument("--learning_rate", default = 1e-1, type=float)
parser.add_argument("--decay_points", default = [.3, .6, .8], type=float, nargs = '+')
parser.add_argument("--decay_rate", default=.2, type=float)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--batch_size", default = 128, type=int)
parser.add_argument("--val_batch_size", default=256, type=int)
parser.add_argument("--train_epoch", default=200, type=int)

parser.add_argument("--pruning", default=False, action = 'store_true')
parser.add_argument("--trained_param", type=str)
parser.add_argument("--target_rate", default=.6, type=float)
parser.add_argument("--search_step", default=.2, type=float)
parser.add_argument("--minimum_rate", default=.0, type=float)
parser.add_argument("--num_teacher", default=5, type=int)
parser.add_argument("--Knowledge", default=False, action = 'store_true')

parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--gpu_id", default= [0], type=int, nargs = '+')
parser.add_argument("--do_log", default=200, type=int)
parser.add_argument("--compile", default=True, action = 'store_true')
parser.add_argument("--setting", default=None, type=str)
args = parser.parse_args()

args.home_path = os.path.dirname(os.path.abspath(__file__))
args.decay_points = [int(dp*args.train_epoch) if dp < 1 else int(dp) for dp in args.decay_points]

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices([tf.config.list_physical_devices('GPU')[i] for i in args.gpu_id], 'GPU')
    for gpu_id in args.gpu_id:
        tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
    devices = ['/gpu:{}'.format(i) for i in args.gpu_id]
    utils.save_code_and_augments(args)
       
    strategy = tf.distribute.MirroredStrategy(devices, cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())
    with strategy.scope():
        datasets = CIFAR.build_dataset_providers(args, strategy)
        model = utils.load_model(args, args.num_classes, args.trained_param )

        train_step, train_loss, train_accuracy, optimizer = op_utils.Optimizer(args, model, strategy )

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction = tf.keras.losses.Reduction.SUM)
        Eval = utils.Evaluation(args, model, strategy, datasets['test'], loss_object)

        print ('Training starts')
        fine_tuning_time = 0
        tic = time.time()

        logs = {'train':[], 'test':[], 'test_loss': []}
        summary_writer = tf.summary.create_file_writer(args.train_path, flush_millis = 30000)
        lr = args.learning_rate

        with summary_writer.as_default():
            for step, data in enumerate(datasets['train']):
                epoch = step//args.iter_len['train']

                lr = utils.scheduler(args, step)
                optimizer.learning_rate = lr

                step += 1
                train_step(*data)
                
                if step % args.do_log == 0:
                    template = 'Global step {0:5d}: loss = {1:0.4f} ({2:1.3f} sec/step)'
                    train_time = time.time() - tic
                    print (template.format(step, train_loss.result()*len(args.gpu_id), train_time/args.do_log))
                    fine_tuning_time += train_time
                    tic = time.time()

                if step % args.iter_len['train'] == 0:
                    tic_ = time.time()
                    test_acc, test_loss = Eval.run(False)
                    utils.save_model(args, model, 'trained_params')

                    tf.summary.scalar('Categorical_loss/train', train_loss.result()*len(args.gpu_id), step=epoch+1)
                    tf.summary.scalar('Categorical_loss/test', test_loss*len(args.gpu_id), step=epoch+1)
                    tf.summary.scalar('Accuracy/train', train_accuracy.result()*100, step=epoch+1)
                    tf.summary.scalar('Accuracy/test', test_acc*100, step=epoch+1)
                    tf.summary.scalar('learning_rate', lr, step=epoch+1)
                    summary_writer.flush()

                    template = 'Epoch: {0:3d}, train_loss: {1:0.4f}, train_Acc.: {2:2.2f}, val_loss: {3:0.4f}, val_Acc.: {4:2.2f}'
                    print (template.format(epoch+1, train_loss.result()*len(args.gpu_id), train_accuracy.result()*100,
                                                    test_loss*len(args.gpu_id),  test_acc*100))
                    logs['train'].append(train_accuracy.result().numpy()*100)
                    logs['test'].append(test_acc*100)
                    logs['test_loss'].append(test_loss)
                    train_loss.reset_states()
                    train_accuracy.reset_states()
                    tic += time.time() - tic_

        utils.save_model(args, model, 'trained_params')
        np.save(os.path.join(args.train_path, 'logs'), logs)
        train_time = time.time() - tic
        fine_tuning_time += train_time
        args.fine_tuning_time = fine_tuning_time
        print ('fine_tuning_time: %f'%fine_tuning_time)

        p, f = utils.check_complexity(model, args)
        args.FLOPS = f
        args.params = p
        args.test_acc = test_acc*100
        utils.save_code_and_augments(args)
