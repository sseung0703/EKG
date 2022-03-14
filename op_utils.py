import tensorflow as tf
import numpy as np
import utils

def Optimizer(args, model, strategy):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction = tf.keras.losses.Reduction.SUM)    
    optimizer = tf.optimizers.SGD(learning_rate = args.learning_rate, momentum = .9, nesterov=True)
        
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    if args.Knowledge:
        def objective(images, labels, K):
            T = 4 if args.num_classes < 1000 else 10

            def kld(x, y, axis = -1, keepdims=True):
                return T*tf.reduce_sum(tf.nn.softmax(x/T, axis)*(tf.nn.log_softmax(x/T, axis) - tf.nn.log_softmax(y/T, axis)), axis, keepdims=keepdims)

            with tf.GradientTape() as tape:
                B,N,H,W,D=images.shape
                images = tf.reshape(images, [-1, H,W,D])
                preds = model(images, training = True)

                preds = tf.reshape(preds, [B, N, -1])
                top = tf.gather(K, 0, axis = -1)
           
                curr_point = tf.reduce_sum(kld(top, tf.reduce_mean(preds, 1), axis = 1))
                mask = tf.cast(tf.reduce_sum(kld(tf.expand_dims(top, -1), K, axis = 1, keepdims=True), 0, keepdims=True) < curr_point, tf.float32)
                top2 = tf.reduce_sum(K * mask,-1) / tf.maximum(1., tf.reduce_sum(mask))

                total_loss = tf.add_n([loss_object(labels, tf.gather(preds, i, axis = 1)) + tf.reduce_sum(kld(top2, tf.gather(preds, i, axis = 1))) for i in range(N)])/args.batch_size

            gradients = [g/2 for g in tape.gradient(total_loss, model.trainable_variables)]
            update_vars = [tf.stack(getattr(L, 'update_var', 0.)) for k, L in model.Layers.items() ]

            for j in range(2):
                train_accuracy.update_state(labels, tf.gather(preds,j,axis=1))

            return total_loss, gradients, update_vars
    else:
        def objective(images, labels):
            with tf.GradientTape() as tape:
                pred = model(images, training = True)
                total_loss = loss_object(labels, pred)/args.batch_size
            gradients = tape.gradient(total_loss, model.trainable_variables)
            update_vars = [tf.stack(getattr(L, 'update_var', 0.)) for k, L in model.Layers.items() ]
            train_accuracy.update_state(labels, pred)
            return total_loss, gradients, update_vars

    if args.accum > 1:
        model(np.zeros([1]+args.input_size), training = True)

    @tf.function(jit_compile = args.compile)
    def compiled_step(*data):
        if args.accum < 2:
            total_loss, gradients, update_vars = objective(*data)
        else:
            total_loss, gradients, update_vars = utils.accumulator(args.batch_size, args.accum, len(args.gpu_id), objective, data, 
                [
                    tf.constant(0.), # loss
                    [tf.zeros_like(v) for v in model.trainable_variables], # gradients
                    [tf.stack([tf.zeros_like(v) for v in L.update_var]) if hasattr(L, 'update_var') else tf.constant(0.) for k, L in model.Layers.items() ], # moving statistics
                ]
            )

        gradients = [g + v * args.weight_decay / len(args.gpu_id) for g, v in zip(gradients, model.trainable_variables)]
        update_vars = [tf.unstack(uv/args.accum) if uv.shape != [] else uv for uv in update_vars]

        return total_loss, gradients, update_vars

    def train_step(*data):
        total_loss, gradients, update_vars = compiled_step(*data)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        for (_, layer), v in zip(model.Layers.items(), update_vars):
            if hasattr(layer, 'update'):
                layer.update(v)
        
        train_loss.update_state(total_loss)
        
    @tf.function
    def train_step_dist(*data):
        strategy.run(train_step, args= (data))
        lr = optimizer._decayed_lr(var_dtype = tf.float32)
        return lr 

    return train_step_dist, train_loss, train_accuracy, optimizer
