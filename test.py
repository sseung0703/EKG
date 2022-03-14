import os, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from dataloader import CIFAR
import utils

home_path = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description='')

parser.add_argument("--arch", default='ResNet-56', type=str)
parser.add_argument("--dataset", default='CIFAR10', type=str)

parser.add_argument("--val_batch_size", default=256, type=int)
parser.add_argument("--trained_param", type=str)
parser.add_argument("--data_path", type=str)

parser.add_argument("--gpu_id", default= [0], type=int, nargs = '+')
parser.add_argument("--compile", default = True, action = 'store_true')

args = parser.parse_args()

args.home_path = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices([tf.config.list_physical_devices('GPU')[i] for i in args.gpu_id], 'GPU')
    for gpu_id in args.gpu_id:
        tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
    devices = ['/gpu:{}'.format(i) for i in args.gpu_id]
    strategy = tf.distribute.MirroredStrategy(devices, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

    with strategy.scope():
        datasets = CIFAR.build_dataset_providers(args, strategy, test_only = True)

        model = utils.load_model(args, args.num_classes, args.trained_param)

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction = tf.keras.losses.Reduction.SUM)
        loss_accum = tf.keras.metrics.Mean(name='loss')
        top1_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='top1_accuracy')
        top5_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_accuracy')

        @tf.function(experimental_compile = args.compile)
        def test_step(images, labels):
            pred = model(images, training = True)
            loss = loss_object(labels, pred)/args.val_batch_size
            loss_accum.update_state(loss)
            top1_accuracy.update_state(labels, pred)
            top5_accuracy.update_state(labels, pred)

        @tf.function(experimental_compile = args.compile)
        def test_step_dist(images, labels):
            strategy.run(test_step, args=(images, labels))

        for i, (test_images, test_labels) in enumerate(datasets['test']):
            test_step_dist(test_images, test_labels)

        loss = loss_accum.result().numpy()
        top1_acc = top1_accuracy.result().numpy()
        top5_acc = top5_accuracy.result().numpy()
        top1_accuracy.reset_states()
        top5_accuracy.reset_states()
        print ('Test loss: %.4f, Test ACC. Top-1: %.4f, Top-5: %.4f'%(loss, top1_acc, top5_acc))
        p, f = utils.check_complexity(model, args)
        print ('Params: %.2f'%(p/1e6), 'FLOPS: %.2f'%(f/1e6))
