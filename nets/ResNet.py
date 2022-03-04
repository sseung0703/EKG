import tensorflow as tf
from . import tcl
from collections import OrderedDict
from functools import partial

class Model(tf.keras.Model):
    def __init__(self, args, num_layers, num_class, name = 'ResNet', trainable = True, **kwargs):
        super(Model, self).__init__(name = name, **kwargs)
        def kwargs(**kwargs):
            return kwargs
        Conv = partial(tcl.Conv2d, use_biases = False, activation_fn = None, trainable = trainable)
        BN   = partial(tcl.BatchNorm, trainable = trainable)
        FC   = partial(tcl.FC, trainable = trainable)

        self.num_layers = num_layers
        self.args = args

        self.Layers = OrderedDict()
        
        network_argments = {
            ## ILSVRC
            18 : {'blocks' : [2,2,2,2],'depth' : [64,128,256,512], 'strides' : [1,2,2,2]},
            34 : {'blocks' : [3,4,6,3],'depth' : [64,128,256,512], 'strides' : [1,2,2,2]},
            50 : {'blocks' : [3,4,6,3],'depth' : [64,128,256,512], 'strides' : [1,2,2,2]},

            ## CIFAR
            56 : {'blocks' : [9,9,9],'depth' : [16,32,64], 'strides' : [1,2,2]},
            20 : {'blocks' : [3,3,3],'depth' : [16,32,64], 'strides' : [1,2,2]},
        }
        self.net_args = network_argments[self.num_layers]

        self.in_to_out= {}

        if num_class == 1000:
            self.Layers['conv'] = Conv([7,7], self.net_args['depth'][0], strides = 2, name = 'conv', layertype = 'input')
            self.Layers['bn']   = BN(name = 'bn')
            self.maxpool_3x3 = tf.keras.layers.MaxPool2D((3,3), strides = 2, padding = 'SAME')

        else:
            self.Layers['conv'] = Conv([3,3], self.net_args['depth'][0], name = 'conv', layertype = 'input')
            self.Layers['bn']   = BN(name = 'bn')

        self.in_to_out['conv'] = ['input']
        out = 'conv'
        residue = 'conv'

        self.expansion = 4 if self.num_layers in {50} else 1
        in_depth = self.net_args['depth'][0]
        for i, (nb_resnet_layers, depth, strides) in enumerate(zip(self.net_args['blocks'], self.net_args['depth'], self.net_args['strides'])):
            for j in range(nb_resnet_layers):
                name = 'BasicBlock%d.%d/'%(i,j)
                if j != 0:
                    strides = 1

                if getattr(self.args, 'pruning', False):
                    self.Layers[name + 'dummy']   = tcl.Dummy(name = name + 'dummy')
                    self.in_to_out[name + 'dummy'] = [out]
                    if strides > 1 or depth * self.expansion != in_depth:
                        self.in_to_out[name + 'dummy'].append(residue)

                if strides > 1 or depth * self.expansion != in_depth:
                    self.Layers[name + 'conv3'] = Conv([1,1], depth * self.expansion, strides = strides, name = name +'conv3')
                    self.Layers[name + 'bn3']   = BN(name = name + 'bn3')
                    self.in_to_out[name + 'conv3'] = []

                if self.num_layers in {18, 56, 20, 34}:
                    self.Layers[name + 'conv1'] = Conv([3,3], depth, strides = strides, name = name + 'conv1')
                    self.Layers[name + 'bn1']   = BN( name = name + 'bn1')
                    self.in_to_out[name + 'conv1'] = []
                    out = name + 'conv1'

                    self.Layers[name + 'conv2'] = Conv([3,3], depth * self.expansion, name = name + 'conv2')
                    self.Layers[name + 'bn2']   = BN( name = name + 'bn2',
                        param_initializers = {'gamma': tf.keras.initializers.Zeros()})
                    self.in_to_out[name + 'conv2'] = [out]
                    out = name + 'conv2'

                else:
                    self.Layers[name + 'conv0'] = Conv([1,1], depth, name = name + 'conv0')
                    self.Layers[name + 'bn0']   = BN( name = name + 'bn0')
                    self.in_to_out[name + 'conv0'] = []
                    out = name + 'conv0'

                    self.Layers[name + 'conv1'] = Conv([3,3], depth, strides = strides, name = name + 'conv1')
                    self.Layers[name + 'bn1']   = BN( name = name + 'bn1')
                    self.in_to_out[name + 'conv1'] = [out]
                    out = name + 'conv1'

                    self.Layers[name + 'conv2'] = Conv([1,1], depth * self.expansion, name = name + 'conv2')
                    self.Layers[name + 'bn2']   = BN( name = name + 'bn2',
                        param_initializers = {'gamma': tf.keras.initializers.Zeros()})
                    self.in_to_out[name + 'conv2'] = [out]
                    out = name + 'conv2'

                if strides > 1 or depth * self.expansion != in_depth:
                    residue = name + 'conv3'

                in_depth = depth * self.expansion

        if getattr(self.args, 'pruning', False):
            self.Layers['dummy']   = tcl.Dummy(name = 'dummy')
            self.in_to_out['dummy'] = [out]

        self.Layers['fc'] = FC(num_class, name = 'fc')

    def call(self, x, training=None):
        x = self.Layers['conv'](x)
        x = self.Layers['bn'](x)
        x = tf.nn.relu(x)
        if hasattr(self, 'maxpool_3x3'):
            x = self.maxpool_3x3(x)

        in_depth = self.net_args['depth'][0]

        for i, (nb_resnet_layers, depth, strides) in enumerate(zip(self.net_args['blocks'], self.net_args['depth'], self.net_args['strides'])):
            for j in range(nb_resnet_layers):
                name = 'BasicBlock%d.%d/'%(i,j)
                if j != 0:
                    strides = 1

                if getattr(self.args, 'pruning', False):
                    x = self.Layers[name + 'dummy'](x)

                if strides > 1 or depth * self.expansion != in_depth:
                    residual = self.Layers[name + 'conv3'](x)
                    residual = self.Layers[name + 'bn3'](residual)
                else:
                    residual = x

                if self.num_layers not in {18, 56, 20, 34}:
                    x = self.Layers[name + 'conv0'](x)
                    x = self.Layers[name + 'bn0'](x)
                    x = tf.nn.relu(x)

                x = self.Layers[name + 'conv1'](x)
                x = self.Layers[name + 'bn1'](x)
                x = tf.nn.relu(x)

                x = self.Layers[name + 'conv2'](x)
                x = self.Layers[name + 'bn2'](x)
                x = tf.nn.relu(x + residual)
                in_depth = depth * self.expansion

        if getattr(self.args, 'pruning', False):
            x = self.Layers['dummy'](x)

        x = tf.reduce_mean(x, [1,2])
        x = self.Layers['fc'](x)

        return x
