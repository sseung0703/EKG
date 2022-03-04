import os, shutil, pickle, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
tf.debugging.set_log_device_placement(False)

from nets import ResNet

def scheduler(args, step):
    epoch = step//args.iter_len['train']
    lr = args.learning_rate

    for dp in args.decay_points:
        if epoch >= dp:
            lr *= args.decay_rate

    return lr

def sorted_idx(x, reverse = False):
    x = np.where(x == 0, x - 1e12, x)
    order = np.lexsort((np.arange(x.shape[0]), np.lexsort((np.arange(x.shape[0]), x))))
    if reverse:
        order = order.shape - order - 1
    return order

def save_code_and_augments(args):
    if os.path.isdir(args.train_path): 
        print ('============================================')
        print ('The folder already is. It will be overwrited')
        print ('============================================')

    else:
        os.mkdir(args.train_path)

    if not(os.path.isdir(os.path.join(args.train_path,'codes'))):
        destination = shutil.copytree(args.home_path, os.path.join(args.train_path,'codes'), copy_function = shutil.copy, 
            ignore = shutil.ignore_patterns('*.pyc','__pycache__','*.swp')) 

    if os.path.isfile(os.path.join(args.train_path, 'arguments.txt')):
        with open(os.path.join(args.train_path, 'arguments.txt')) as json_file:
            args_prev = json.load(json_file)

        args = args.__dict__
        keys = list(set(args) | set(args_prev))
        args = {k: args[k] if k in args else args_prev[k] for k in keys}
        with open(os.path.join(args['train_path'], 'arguments.txt'), 'w') as f:
            json.dump(args, f, indent=2)
    else:
        with open(os.path.join(args.train_path, 'arguments.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

class Evaluation:
    def __init__(self, args, model, strategy, dataset, loss_object):
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        @tf.function(experimental_compile=args.compile)
        def compiled_step(images, labels, training):
            pred = model(images, training = training)
            loss = loss_object(labels, pred)/args.val_batch_size
            return pred, loss

        def eval_step(images, labels, training):
            pred, loss = compiled_step(images, labels, training)
            self.test_loss.update_state(loss)
            self.test_accuracy.update_state(labels, pred)

        @tf.function
        def eval_step_dist(images, labels, training):
            strategy.run(eval_step, args=(images, labels, training))

        self.dataset = dataset
        self.step = eval_step_dist

    def run(self, training):
        for images, labels in self.dataset:
            self.step(images, labels, training)
        loss = self.test_loss.result().numpy()
        acc = self.test_accuracy.result().numpy()
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()
        return acc, loss

def load_model(args, num_class, trained_param = None):
    arch = int(args.arch.split('-')[1])
    model = ResNet.Model(args, num_layers = arch, num_class = num_class, name = 'ResNet', trainable = True)

    if trained_param is not None:
        with open(trained_param, 'rb') as f:
            trained = pickle.load(f)
            assign_param(model, trained)
    return model

def assign_param(model, trained):
    n = 0
    for k in model.Layers.keys():
        layer = model.Layers[k]
        if 'conv' in k or 'fc' in k:
            kernel = trained[layer.name + '/kernel:0']
            layer.kernel_initializer = tf.constant_initializer(kernel)
            n += 1
            if layer.use_biases:
                layer.biases_initializer = tf.constant_initializer(trained[layer.name + '/biases:0'])
                n += 1
            layer.num_outputs = kernel.shape[-1]
            
        elif 'bn' in k:
            moving_mean = trained[layer.name + '/moving_mean:0']
            moving_variance = trained[layer.name + '/moving_variance:0']
            param_initializers = {'moving_mean' : tf.constant_initializer(moving_mean),
                                  'moving_variance': tf.constant_initializer(moving_variance)}
            n += 2

            if layer.scale:
                param_initializers['gamma'] = tf.constant_initializer(trained[layer.name + '/gamma:0'])
                n += 1
            if layer.center:
                param_initializers['beta'] = tf.constant_initializer(trained[layer.name + '/beta:0'])
                n += 1
            layer.param_initializers = param_initializers
    print (n, 'params loaded')

def save_model(args, model, name):
    params = {}
    for v in model.variables:
        if model.name in v.name and 'dummy' not in v.name:
            params[v.name[len(model.name)+1:]] = v.numpy()
    with open(os.path.join(args.train_path, name + '.pkl'), 'wb') as f:
        pickle.dump(params, f)

def check_complexity(model, args):
    model(np.zeros([1]+args.input_size, dtype=np.float32), training = False)
    total_params = []
    total_flops = []
    for k in model.Layers.keys():
        layer = model.Layers[k]
        if hasattr(layer, 'params'):
            p = layer.params
            if isinstance(p, tf.Tensor):
                p = p.numpy()
            total_params.append(p)
        if hasattr(layer, 'flops'):
            f = layer.flops
            if isinstance(f, tf.Tensor):
                f = f.numpy()
            total_flops.append(f)
    return sum(total_params), sum(total_flops)