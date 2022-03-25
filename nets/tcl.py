import tensorflow as tf
import numpy as np

class Conv2d(tf.keras.layers.Layer):
    def __init__(self, kernel_size, num_outputs, strides = 1, dilations = 1, padding = 'SAME',
                 kernel_initializer = tf.keras.initializers.VarianceScaling(scale = 2., mode='fan_out'),
                 use_biases = True,
                 biases_initializer  = tf.keras.initializers.Zeros(),
                 activation_fn = None,
                 name = 'conv',
                 trainable = True,
                 **kwargs):
        super(Conv2d, self).__init__(name = name, trainable = trainable, **kwargs)
        
        self.kernel_size = kernel_size
        self.num_outputs = num_outputs
        self.strides = strides
        self.padding = padding
        self.dilations = dilations
        self.kernel_initializer = kernel_initializer
        
        self.use_biases = use_biases
        self.biases_initializer = biases_initializer
        
        self.activation_fn = activation_fn
        
    def build(self, input_shape):
        super(Conv2d, self).build(input_shape)
        self.kernel = self.add_weight(name  = 'kernel', 
                                      shape = self.kernel_size + [input_shape[-1], self.num_outputs],
                                      initializer = self.kernel_initializer,
                                      trainable = self.trainable)
        if self.use_biases:
            self.biases  = self.add_weight(name = "biases",
                                           shape=[1,1,1,self.num_outputs],
                                           initializer = self.biases_initializer,
                                           trainable = self.trainable)
        self.ori_shape = self.kernel.shape

    def call(self, input):
        kernel = self.kernel
        kh,kw,Di,Do = tf.unstack(tf.cast(tf.shape(kernel), tf.float32))

        mask = 1
        if hasattr(self, 'in_mask'):
            in_mask = self.in_mask.get_mask()
            Di = tf.reduce_sum(in_mask)
            mask = mask * tf.reshape(in_mask, [1,1,-1,1])
            
        if hasattr(self, 'out_mask'):
            out_mask = self.out_mask.get_mask()
            Do = tf.reduce_sum(out_mask)
            mask = mask * tf.reshape(out_mask, [1,1,1,-1])
            
        if isinstance(mask, tf.Tensor):
            norm = tf.linalg.norm(kernel)
            kernel =  kernel*mask
            kernel = tf.linalg.l2_normalize(kernel)*norm

        conv = tf.nn.conv2d(input, kernel, self.strides, self.padding,
                            dilations=self.dilations, name=None)
        
        if self.use_biases:
            conv += self.biases

        if self.activation_fn:
            conv = self.activation_fn(conv)

        H,W = tf.unstack(tf.cast(tf.shape(conv), tf.float32))[1:3]
        self.params = kh*kw*Di*Do
        self.flops  = H*W*self.params
        
        if self.use_biases:
            self.params += Do

        return conv

class DepthwiseConv2d(tf.keras.layers.Layer):
    def __init__(self, kernel_size, multiplier = 1, strides = [1,1,1,1], dilations = [1,1], padding = 'SAME',
                 kernel_initializer = tf.keras.initializers.VarianceScaling(scale = 2., mode='fan_in'),
                 use_biases = True,
                 biases_initializer = tf.keras.initializers.Zeros(),
                 activation_fn = None,
                 name = 'conv',
                 trainable = True,
                 **kwargs):
        super(DepthwiseConv2d, self).__init__(name = name, trainable = trainable, **kwargs)
        
        self.kernel_size = kernel_size
        self.strides = strides if isinstance(strides, list) else [1, strides, strides, 1]
        self.padding = padding
        self.dilations = dilations if isinstance(dilations, list) else [dilations, dilations]
        self.multiplier = multiplier
        self.kernel_initializer = kernel_initializer
        
        self.use_biases = use_biases
        self.biases_initializer = biases_initializer
        
        self.activation_fn = activation_fn
        
    def build(self, input_shape):
        super(DepthwiseConv2d, self).build(input_shape)
        self.kernel = self.add_weight(name  = 'kernel', 
                                      shape = self.kernel_size + [input_shape[-1], self.multiplier],
                                      initializer=self.kernel_initializer,
                                      trainable = self.trainable)
        if self.use_biases:
            self.biases  = self.add_weight(name = "biases",
                                           shape=[1,1,1, input_shape[-1]*self.multiplier],
                                           initializer = self.biases_initializer,
                                           trainable = self.trainable)
        self.ori_shape = self.kernel.shape

    def call(self, input):
        kernel = self.kernel
        kh,kw,Di,Do = kernel.shape

        if hasattr(self, 'in_mask'):
            in_mask = self.in_mask.get_mask()
            Di = tf.reduce_sum(in_mask)
            mask = tf.reshape(in_mask, [1,1,-1,1])
            
            norm = tf.linalg.norm(kernel)
            kernel =  kernel*mask
            kernel = tf.linalg.l2_normalize(kernel)*norm

        conv = tf.nn.depthwise_conv2d(input, kernel, strides = self.strides, padding = self.padding, dilations=self.dilations)

        if self.use_biases:
            conv += self.biases
        if self.activation_fn:
            conv = self.activation_fn(conv)

        H,W = conv.shape[1:3]

        self.params = kh*kw*Di*Do
        self.flops  = H*W*self.params
        
        if self.use_biases:
            self.params += Do

        return conv

class FC(tf.keras.layers.Layer):
    def __init__(self, num_outputs, 
                 kernel_initializer = tf.keras.initializers.random_normal(stddev = 1e-2),
                 use_biases = True,
                 biases_initializer  = tf.keras.initializers.Zeros(),
                 activation_fn = None,
                 name = 'fc',
                 trainable = True, **kwargs):
        super(FC, self).__init__(name = name, trainable = trainable, **kwargs)
        self.num_outputs = num_outputs
        self.kernel_initializer = kernel_initializer
        
        self.use_biases = use_biases
        self.biases_initializer = biases_initializer
        
        self.activation_fn = activation_fn

    def build(self, input_shape):
        super(FC, self).build(input_shape)
        self.kernel = self.add_weight(name  = 'kernel', 
                                      shape = [int(input_shape[-1]), self.num_outputs],
                                      initializer=self.kernel_initializer,
                                      trainable = self.trainable)
        if self.use_biases:
            self.biases  = self.add_weight(name = "biases",
                                           shape=[1,self.num_outputs],
                                           initializer = self.biases_initializer,
                                           trainable = self.trainable)
        self.ori_shape = self.kernel.shape

    def call(self, input):
        kernel = self.kernel
        Di,Do = tf.unstack(tf.cast(tf.shape(kernel), tf.float32))

        if hasattr(self, 'in_mask'):
            in_mask = self.in_mask.get_mask()
            Di = tf.reduce_sum(in_mask)
            mask = tf.reshape(in_mask, [-1,1])
            
            norm = tf.linalg.norm(kernel)
            kernel =  kernel*mask
            kernel = tf.linalg.l2_normalize(kernel)*norm

        fc = tf.matmul(input, kernel)
        
        if self.use_biases:
            fc += self.biases
        if self.activation_fn:
            fc = self.activation_fn(fc)

        self.params = Di*Do         
        self.flops  = self.params
        for n in fc.shape[1:-1]:
            self.flops *= n 
        
        if self.use_biases:
            self.params += Do

        return fc

class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, param_initializers = None,
                       scale = True,
                       center = True,
                       alpha = 0.9,
                       epsilon = 1e-5,
                       activation_fn = None,
                       name = 'bn',
                       trainable = True,
                       **kwargs):
        super(BatchNorm, self).__init__(name = name, trainable = trainable, **kwargs)
        if param_initializers == None:
            param_initializers = {}
        if not(param_initializers.get('moving_mean')):
            param_initializers['moving_mean'] = tf.keras.initializers.Zeros()
        if not(param_initializers.get('moving_variance')):
            param_initializers['moving_variance'] = tf.keras.initializers.Ones()
        if not(param_initializers.get('gamma')) and scale:
            param_initializers['gamma'] = tf.keras.initializers.Ones()
        if not(param_initializers.get('beta')) and center:
            param_initializers['beta'] = tf.keras.initializers.Zeros()
        
        self.param_initializers = param_initializers
        self.scale = scale
        self.center = center
        self.alpha = alpha
        self.epsilon = epsilon
        self.activation_fn = activation_fn

    def build(self, input_shape):
        super(BatchNorm, self).build(input_shape)
        self.moving_mean = self.add_weight(name  = 'moving_mean', trainable = False,
                                      shape = [1]*(len(input_shape)-1)+[int(input_shape[-1])],
                                      initializer=self.param_initializers['moving_mean'],
                                      aggregation=tf.VariableAggregation.MEAN,
                                      )
        self.moving_variance = self.add_weight(name  = 'moving_variance', trainable = False,
                                      shape = [1]*(len(input_shape)-1)+[int(input_shape[-1])],
                                      initializer=self.param_initializers['moving_variance'],
                                      aggregation=tf.VariableAggregation.MEAN,
                                      )
        self.gamma = self.add_weight(name  = 'gamma', 
                                        shape = [1]*(len(input_shape)-1)+[int(input_shape[-1])],
                                        initializer=self.param_initializers['gamma'],
                                        trainable = self.trainable) if self.scale else 1.
        self.beta = self.add_weight(name  = 'beta', 
                                    shape = [1]*(len(input_shape)-1)+[int(input_shape[-1])],
                                    initializer=self.param_initializers['beta'],
                                    trainable = self.trainable) if self.center else 0.

        self.ori_shape = self.moving_mean.shape[-1]
           
    def EMA(self, variable, value):
        update_delta = (variable - value) * (1-self.alpha)
        variable.assign_sub(update_delta)

    def update(self, mean, var):
        self.EMA(self.moving_mean, mean)
        self.EMA(self.moving_variance, var)
        
    def call(self, input, training=None):
        if training:
            mean, var = tf.nn.moments(input, list(range(len(input.shape)-1)), keepdims=True)
            if not( hasattr(self, 'out_mask')):
                self.update(mean, var)
        
        else:
            mean = self.moving_mean
            var = self.moving_variance
            
        gamma, beta = self.gamma, self.beta
        Do = tf.unstack(tf.cast(tf.shape(self.moving_mean), tf.float32))[-1]

        bn = tf.nn.batch_normalization(input, mean, var, offset = beta, scale = gamma, variance_epsilon = self.epsilon)

        if hasattr(self, 'out_mask'):
            Do = tf.reduce_sum(self.out_mask.get_mask(), -1)

        if self.activation_fn:
            bn = self.activation_fn(bn)

        B,*_, D = tf.unstack( tf.cast(tf.shape(bn), tf.float32) )
        S = tf.cast(tf.size(bn), tf.float32) / B / D
        self.params = Do * (2 + self.scale + self.center)
        self.flops  = S * Do * (1 + self.scale)
        
        return bn

class scoring_layer(tf.keras.layers.Layer):
    def __init__(self, shape, name = '', **kwargs):
        super(scoring_layer, self).__init__(name = name, **kwargs)
        self.shape = shape
        self.num_call = 1

        self.score = self.add_weight(name  = 'score', 
                                 shape = [1,1,1,self.shape],
                                 initializer=tf.keras.initializers.Zeros(),
                                 trainable = False)

        self.order = self.add_weight(name  = 'order', 
                                 shape = [1,1,1,self.shape],
                                 initializer=tf.keras.initializers.Zeros(),
                                 trainable = False)

        self.rate = self.add_weight(name  = 'rate',
                                 shape = [],
                                 initializer=tf.keras.initializers.Ones(),
                                 trainable = False)

    def assign_order(self):
        self.order.assign( tf.cast(tf.argsort(tf.argsort( self.score , direction = 'DESCENDING'), direction = 'ASCENDING'), tf.float32) )
    
    def get_mask(self):
        return tf.cast( self.order < tf.maximum(self.shape * self.rate, 1.) , tf.float32)

    def __call__(self, x):
        mask = self.get_mask()
        
        @tf.custom_gradient
        def by_pass(x, score):
            y = x * mask
            return y, lambda dy : [ dy, tf.abs(tf.reduce_sum(dy * y, [0,1,2], keepdims=True)) ]
        return by_pass(x, self.score)
