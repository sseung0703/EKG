import os, shutil, pickle, json, math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from collections import OrderedDict

import numpy as np
import tensorflow as tf

from nets import ResNet

def scheduler(learning_rate, epoch, decay_points, decay_rate):
    lr = learning_rate

    for dp in decay_points:
        if epoch >= dp:
            lr *= decay_rate

    return lr

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
            args_prev = json.load(json_file, object_pairs_hook=OrderedDict)

        args = OrderedDict(args.__dict__)
        for a, v in args.items():
            if a not in args_prev:
                args[a] = v
        with open(os.path.join(args['train_path'], 'arguments.txt'), 'w') as f:
            json.dump(OrderedDict(args), f, indent=2)
    else:
        with open(os.path.join(args.train_path, 'arguments.txt'), 'w') as f:
            json.dump(OrderedDict(args.__dict__), f, indent=2)

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
    if 'ResNet' in args.arch:
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
        if model.name in v.name:
            params[v.name[len(model.name)+1:]] = v.numpy()
    with open(os.path.join(args.train_path, name + '.pkl'), 'wb') as f:
        pickle.dump(params, f)

def check_complexity(model, args):
    model(np.zeros([1]+args.input_size, dtype=np.float32), training = False)
    total_params = []
    total_flops = []

    total_flops = sum([np.mean(layer.flops.numpy()) for _, layer in model.Layers.items() if hasattr(layer, 'flops')])
    total_params = sum([np.mean(layer.params.numpy()) for _, layer in model.Layers.items() if hasattr(layer, 'params')])
    return total_params, total_flops

def accumulator(batch_size, accum_num, num_gpu, graph, inputs, outputs):
    b = batch_size // accum_num
    indices = tf.expand_dims(tf.range(batch_size), -1)
    
    i = tf.constant(0)
    c = lambda i, *o : (tf.less(i, accum_num))
    def accum_loop(i, *outputs):
        def mapper(X):
            if X.shape[0] == b:
                return tf.slice(X, [b*i]+[0]*(len(X.shape)-1), [b, *X.shape[1:]] )
            else:
                return X
        input_split = [[mapper(x) for x in X] if isinstance(X, list) else mapper(X) for X in inputs]

        o = graph(*input_split)

        def mapper(X, x):
            if X.shape == x.shape:
                return X + x
            else:
                return tf.tensor_scatter_nd_update(X, tf.slice(indices, [b*i,0],[b,1]), x)
        return (i+1, *[ [mapper(*o__) for o__ in zip(*o_) ] if isinstance(o_[0], list) else mapper(*o_)  for o_ in zip(outputs, o)])
    return tf.while_loop(c, accum_loop, [i, *outputs])[1:]
