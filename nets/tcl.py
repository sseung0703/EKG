import tensorflow as tf
import numpy as np

def Proposed(x, dx):
    dkernel = tf.abs( tf.reduce_sum( x * dx, [0,1,2] ) )
    return dkernel

class Conv2d(tf.keras.layers.Layer):
    def __init__(self, kernel_size, num_outputs, strides = 1, dilations = 1, padding = 'SAME',
                 kernel_initializer = tf.keras.initializers.VarianceScaling(scale = 2., mode='fan_out'),
                 use_biases = True,
                 biases_initializer  = tf.keras.initializers.Zeros(),
                 activation_fn = None,
                 name = 'conv',
                 trainable = True,
                 layertype = 'mid',
                 keep_feat = False,
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

        self.type = layertype
        self.keep_feat = keep_feat
        self.scoring = False
        
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
        kh,kw,Di,Do = kernel.shape

        if hasattr(self, 'in_depth'):
            Di = tf.math.ceil(self.ori_shape[2]*self.in_depth)

        if hasattr(self, 'out_depth'):
            Do = tf.math.ceil(self.ori_shape[3]*self.out_depth)

        if not(self.scoring):
            conv = tf.nn.conv2d(input, kernel, self.strides, self.padding,
                                dilations=self.dilations, name=None)
        else:
            @tf.custom_gradient
            def scoring(x, kernel):
                B,H,W,D = x.shape
                out_mask = tf.cast(tf.less(self.out_mask, Do),tf.float32)
                out_mask = tf.reshape(out_mask, [1,1,1,-1])
                in_mask = tf.cast(tf.less(self.in_mask, Di),tf.float32)
                in_mask = tf.reshape(in_mask, [1,1,-1,1])

                norm = tf.linalg.norm(kernel)
                kernel =  kernel*in_mask*out_mask
                kernel = tf.linalg.l2_normalize(kernel)*norm

                y = tf.nn.conv2d(x, kernel, self.strides, self.padding, dilations = self.dilations, name=None)

                def gradient_scoring(dy):
                    dx = tf.gradients(y, x, dy)[0] * tf.transpose(in_mask, [0,1,3,2])
                    dkernel = Proposed(x, dx)
                    return [dx, dkernel]
                return y, gradient_scoring

            conv = scoring(input, kernel)

        if self.use_biases:
            conv += self.biases

        if self.activation_fn:
            conv = self.activation_fn(conv)

        H,W = conv.shape[1:3]
        self.params = kh*kw*Di*Do
        self.flops  = H*W*self.params
        
        if self.use_biases:
            self.params += Do
        if self.keep_feat:
            self.feat = conv
        return conv

class Dummy(tf.keras.layers.Layer):
    def __init__(self, name = 'dummy', **kwargs):
        super(Dummy, self).__init__(name = name, **kwargs)
        self.get_score = False
        self.scoring = False
        
    def build(self, input_shape):
        super(Dummy, self).build(input_shape)
        self.kernel = self.add_weight(name  = 'kernel', shape = [], trainable = False)

    def call(self, input):
        @tf.custom_gradient
        def scoring(x, kernel):
            def gradient_scoring(dy):
                dx = dy
                dkernel = Proposed(x, dx)
                return [dx, dkernel]
            return x, gradient_scoring
        output = scoring(input, self.kernel)
        return output

class FC(tf.keras.layers.Layer):
    def __init__(self, num_outputs, 
                 kernel_initializer = tf.keras.initializers.random_normal(stddev = 1e-2),
                 use_biases = True,
                 biases_initializer  = tf.keras.initializers.Zeros(),
                 activation_fn = None,
                 name = 'fc',
                 keep_feat = False,
                 trainable = True, **kwargs):
        super(FC, self).__init__(name = name, trainable = trainable, **kwargs)
        self.num_outputs = num_outputs
        self.kernel_initializer = kernel_initializer
        
        self.use_biases = use_biases
        self.biases_initializer = biases_initializer
        
        self.activation_fn = activation_fn
        
        self.keep_feat = keep_feat
        self.scoring = False
        
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
        Di,Do = kernel.shape

        if hasattr(self, 'in_depth'):
            Di = tf.math.ceil(self.ori_shape[0]*self.in_depth)

        if not(self.scoring):
            fc = tf.matmul(input, kernel)
        else:
            @tf.custom_gradient
            def scoring(x, kernel):
                in_mask = tf.cast(tf.less(self.in_mask, Di),tf.float32)
                in_mask = tf.reshape(in_mask, [-1,1])
                
                norm = tf.linalg.norm(kernel)
                kernel =  kernel*in_mask
                kernel = tf.linalg.l2_normalize(kernel)*norm


                y = tf.matmul(input, kernel)

                def gradient_scoring(dy):
                    dx = tf.gradients(y, x, dy)[0]
                    dkernel = Proposed(x, dx)
                    return [dx, dkernel]
                return y, gradient_scoring
            fc = scoring(input, kernel)

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
        if self.keep_feat:
            self.in_feat = input
            self.feat = fc
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
                       keep_feat = False,
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
        self.keep_feat = keep_feat
        self.scoring = False

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
        if self.scale:
            self.gamma = self.add_weight(name  = 'gamma', 
                                         shape = [1]*(len(input_shape)-1)+[int(input_shape[-1])],
                                         initializer=self.param_initializers['gamma'],
                                         trainable = self.trainable)
        else:
            self.gamma = 1.
        if self.center:
            self.beta = self.add_weight(name  = 'beta', 
                                        shape = [1]*(len(input_shape)-1)+[int(input_shape[-1])],
                                        initializer=self.param_initializers['beta'],
                                        trainable = self.trainable)
        else:
            self.beta = 0.
        self.ori_shape = self.moving_mean.shape[-1]
           
    def EMA(self, variable, value):
        update_delta = (variable - value) * (1-self.alpha)
        variable.assign_sub(update_delta)

    def update(self, update_var):
        mean, var = update_var
        self.EMA(self.moving_mean, mean)
        self.EMA(self.moving_variance, var)
        
    def call(self, input, training=None):
        if training:
            mean, var = tf.nn.moments(input, list(range(len(input.shape)-1)), keepdims=True)
            self.update_var = [mean, var]
        
        else:
            mean = self.moving_mean
            var = self.moving_variance
            
        gamma, beta = self.gamma, self.beta
        Do = self.moving_mean.shape[-1]

        if hasattr(self, 'out_depth'):
            Do = tf.math.ceil(self.ori_shape*self.out_depth)
            out_mask = tf.cast(tf.less(self.out_mask, Do),tf.float32)
            out_mask = tf.reshape(out_mask, [1]*(len(input.shape)-1)+[-1])
            gamma = gamma * out_mask
            beta = beta * out_mask

            if not(self.scoring):
                bn = tf.nn.batch_normalization(input, mean, var, offset = beta, scale = gamma, variance_epsilon = self.epsilon)

            else:
                @tf.custom_gradient
                def scoring(x, mean, var, gamma, beta):
                    y = tf.nn.batch_normalization(x, mean, var, offset = beta, scale = gamma, variance_epsilon = self.epsilon)

                    def gradient_scoring(dy):
                        B,H,W,D = dy.shape
                        dx = tf.gradients(y, x, dy)[0]
                        dgamma = tf.abs(tf.reduce_sum( y * dy, [0,1,2], keepdims=True))
                        return [dx, None, None, dgamma, None]
                    return y, gradient_scoring
                bn = scoring(input, mean, var, gamma, beta)

        else:
            bn = tf.nn.batch_normalization(input, mean, var, offset = beta, scale = gamma, variance_epsilon = self.epsilon)

        if self.activation_fn:
            bn = self.activation_fn(bn)
        self.params = Do * (2 + self.scale + self.center)
        self.flops  = self.params
        for n in bn.shape[1:-1]:
            self.flops *= n
        if self.keep_feat:
            self.feat = bn
        return bn