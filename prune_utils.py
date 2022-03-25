import time

import tensorflow as tf
import numpy as np
from math import ceil
from tqdm import tqdm
                                 
import utils
from nets import tcl
from dataloader import CIFAR

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction = tf.keras.losses.Reduction.SUM)
ce_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction = tf.keras.losses.Reduction.SUM)

class EKG:
    def __init__(self, args, model, strategy, datasets):
        self.args = args
        self.model = model
        self.strategy = strategy

        self.val_ds_ori = datasets['val']
        self.eval_ds = datasets['val'].batch(self.args.val_batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        self.initialize_Gate(args, model)
        self.define_score_step()
        self.define_eval_step()

        self.width_list = [1.] * len(self.Gates)
        self.width_list_temp = self.width_list.copy()

        (self.cur_p, self.cur_f) = (self.ori_p, self.ori_f) = utils.check_complexity(model, args)
        self.update_Knowledge(init_Knowledge=True)

        self.step = 0

    def run(self):
        summary_writer = tf.summary.create_file_writer(self.args.train_path)
        with summary_writer.as_default():
            tf.summary.scalar('Greed_search/acc', self.cur_acc, step=self.step)
            tf.summary.scalar('Greed_search/loss', self.cur_loss, step=self.step)
            tf.summary.scalar('Greed_search/flops_rate', self.cur_f/self.ori_f, step=self.step)
            summary_writer.flush()
            
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            
            while (self.cur_f/self.ori_f > self.args.search_target_rate):
                knowledge = self.search_step()
                self.update_Knowledge(knowledge)
                summary_writer.flush()

    def search_step(self):
        # search next steps
        self.get_score()
        do_check = self.mask_marking()

        accuracy_list = np.zeros([self.num_gate])
        losses_list = np.ones([self.num_gate]) * 1e12
        scores_list = np.ones([self.num_gate]) * 1e12
        knowledge_list = [0]*self.num_gate
        
        for i in tqdm(range(self.num_gate)):
            if self.width_list[i] > 0 and do_check[i]:
                knowledge_list[i], accuracy_list[i], losses_list[i], scores_list[i] = self.check_next_step(i)

        # assign the best step
        best_idx = np.argmin( scores_list )
            
        self.assign_new_width(losses_list, accuracy_list, best_idx)
        return knowledge_list[best_idx]

    def set_width(self, width_list):
        for r, w in zip(self.rate_var, width_list):
            r.assign(w)
        return utils.check_complexity(self.model, self.args)

    def fine_search(self, i):
        p, f = self.set_width(self.width_list)
        W = self.width_list[i]

        for w in np.arange(W, W + self.args.search_step, self.fine_search_step_list[i]):
            self.width_list[i] = round(w, 2)
            p, f = self.set_width(self.width_list)

            if f/self.ori_f > self.args.search_target_rate:
                self.width_list[i] = round(w - self.fine_search_step_list[i], 5)
                p, f = self.set_width(self.width_list)
                break

        return p, f

    def check_next_step(self, i):
        self.width_list_temp[i] = self.width_list[i]

        self.width_list[i] = round(self.width_list[i] - self.search_step_list[i], 5)
        p, f = self.set_width(self.width_list)
        if f/self.ori_f < self.args.search_target_rate:
            p, f = self.fine_search(i)

        Knowledge, acc, loss, score = self.Eval()

        self.width_list_temp[i], self.width_list[i] = self.width_list[i], self.width_list_temp[i]
        return Knowledge, acc, loss, score

    def assign_new_width(self, losses_list, accuracy_list, idx):
        if not(hasattr(self, 'width_history')):
            self.width_history = [ [ [o.numpy() for o in self.order_var] + [r.numpy() for r in self.rate_var], self.cur_loss.copy()] ]

        self.width_list[idx] = self.width_list_temp[idx]
        self.cur_p, self.cur_f = self.set_width(self.width_list)
        self.cur_acc = accuracy_list[idx]
        self.prev_loss, self.cur_loss = self.cur_loss, losses_list[idx]

        self.step += 1
        print ('Step: %d'%self.step)
        print (self.width_list, idx)
        print ('Ori Acc.: %.2f, Current Acc.: %.2f Ori Loss.: %.4f, Current Loss.: %.4f'
            %(100*self.ori_acc, 100*self.cur_acc, self.ori_loss, self.cur_loss))
        print ('Ori Param.: %.4fM, Slim Param.: %.4fM, Ori FLOPS: %.4fM, Slim FLOPS: %.4fM, Curr. rate: %.2f\n'
            %(self.ori_p/1e6, self.cur_p/1e6, self.ori_f/1e6, self.cur_f/1e6, self.cur_f/self.ori_f*100))

        tf.summary.scalar('Greed_search/acc', self.cur_acc, step=self.step)
        tf.summary.scalar('Greed_search/loss', self.cur_loss, step=self.step)
        tf.summary.scalar('Greed_search/flops_rate', self.cur_f/self.ori_f, step=self.step)

        self.width_history.append([[o.numpy() for o in self.order_var] + [r.numpy() for r in self.rate_var], self.cur_loss.copy()])
        
    def trimed_otsu(self, scores, cut = .2):
        scores = np.sort(scores)
        trim = int(ceil(len(scores) * cut))
        scores = scores[trim:-trim]
        val = []
        for t in range(1,len(scores)-1):
            w0 = t
            w1 = len(scores) - t
            m1 = np.mean(scores[:t])
            m2 = np.mean(scores[t:])
            val.append(w0 * w1 * (m1 - m2)**2)
        return (scores[np.argmax(val)] + scores[np.argmax(val) + 1])/2

    def mask_marking(self):
        scores = []
        
        for r, s, ss in zip(self.rate_var, self.score_var, self.search_step_list):
            r = r.numpy()
            s = s.numpy()

            s = np.sort(s)
            if r > 0:
                r = 1. - r
                s = s[round(r * s.size):round((r + ss) * s.size)]
                scores.append(np.sum(s))
            else:
                scores.append(1e12)
        score_th = self.trimed_otsu([s for s in scores if s < 1e12])
        do_check = np.array(scores) <= score_th
        print ('# of layers to evaluated is %d/%d (%.2f)'%(np.sum(do_check), len(do_check), np.sum(do_check)/len(do_check) * 100 ))
        return do_check

    def initialize_Gate(self, args, model):
        Gate = []
        if 'ResNet' in args.arch:
            for k, layer in model.Layers.items():
                if 'conv' in k:
                    if 'conv2' in k:
                        layer.in_mask = in_mask
                        layer.out_mask = group_mask
                        in_mask = group_mask
                        group_mask.num_call += 1

                    else:
                        layer.out_mask = tcl.scoring_layer(layer.kernel.shape[3], name = layer.name)
                        Gate.append(layer.out_mask)

                        if k == 'conv':
                            group_mask = layer.out_mask
                            in_mask = layer.out_mask

                        elif 'conv3' in k:
                            group_mask = layer.out_mask
                            layer.in_mask = in_mask

                        else:
                            layer.in_mask = in_mask
                            in_mask = layer.out_mask

                    if k.replace('conv', 'bn') in model.Layers:
                        model.Layers[k.replace('conv', 'bn')].out_mask = layer.out_mask

                if 'fc' in k:
                    layer.in_mask = in_mask
        
        self.Gates = Gate
        self.num_gate = len(self.Gates)
        self.score_var = [gate.score for gate in self.Gates]
        self.order_var = [gate.order for gate in self.Gates]
        self.rate_var = [gate.rate for gate in self.Gates]

        self.search_step_list = [max(1/gate.shape, args.search_step/gate.num_call ) for gate in self.Gates]
        self.fine_search_step_list = [1/gate.shape for gate in self.Gates]

    def objective(self, images, labels, K):
        pred = self.model(images, training = True)
        loss = loss_object(labels, pred)        
        score = loss + ce_object(tf.nn.softmax(K), pred)

        loss /= self.args.val_batch_size
        score /= self.args.val_batch_size
        return pred, loss, score

    ## Define evaluation step
    def define_score_step(self):
        def objective_grad(images, labels, K):
            with tf.GradientTape(watch_accessed_variables = False) as tape:
                tape.watch(self.score_var)
                score = self.objective(images, labels, K)[-1]
            grad = tape.gradient(score, self.score_var)
            return grad

        @tf.function(jit_compile = self.args.compile)
        def score_step(*data):
            if self.args.accum < 2:
                gradients = objective_grad(*data)
            else:
                gradients = utils.accumulator(data[0].shape[0], self.args.accum, len(self.args.gpu_id), objective_grad, data, 
                    [tf.zeros_like(v) for v in self.score_var], # gradients                    
                )
            for s, g in zip(self.score_var, gradients):
                s.assign_add(g)

        @tf.function
        def dist_score_step(*data):
            self.strategy.run(score_step, args=(data))

        self.score_step = dist_score_step

    def get_score(self):
        t = time.time()
        for s in self.score_var:
            s.assign(tf.zeros_like(s))

        for data in self.val_ds:
            self.score_step(*data)
        
        for gate in self.Gates:
            gate.assign_order()
        
        print ('Importance searching time: %.2f'%(time.time()-t))

    ## Define evaluation step
    def define_eval_step(self):
        self.loss_accum = tf.keras.metrics.Mean(name='loss')
        self.score_accum = tf.keras.metrics.Mean(name='score')
        self.acc_accum = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

        @tf.function(jit_compile = self.args.compile)
        def eval_step(*data):
            if self.args.accum < 2:
                pred, loss, score = self.objective(*data)
            else:
                pred, loss, score = utils.accumulator(data[0].shape[0], self.args.accum, len(self.args.gpu_id), self.objective, data, 
                    [
                        tf.zeros([data[0].shape[0], self.args.num_classes]), # pred
                        tf.constant(0.), # loss
                        tf.constant(0.), # score
                    ]
                )
            self.acc_accum.update_state(data[1], pred)
            self.loss_accum.update_state(loss)
            self.score_accum.update_state(score)
            return pred

        @tf.function
        def dist_eval_step(*data):
            k = tf.concat(self.strategy.experimental_local_results(self.strategy.run(eval_step, args=(data))), 0)
            return k

        self.eval_step = dist_eval_step

    def Eval(self):
        Knowledge = np.concatenate([self.eval_step(*data) for data in self.val_ds])
        acc = self.acc_accum.result().numpy()
        loss = self.loss_accum.result().numpy()
        score = self.score_accum.result().numpy()
        self.acc_accum.reset_states()
        self.loss_accum.reset_states()
        self.score_accum.reset_states()
        return Knowledge, acc, loss, score

    def update_Knowledge(self, Knowledge = None, init_Knowledge = False):
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        if init_Knowledge:
            self.Knowledge = np.zeros([self.args.cardinality['val'], self.args.num_classes], dtype = np.float32) 
            Knowledge = tf.data.Dataset.from_tensor_slices(self.Knowledge)
            val_ds = tf.data.Dataset.zip((self.val_ds_ori, Knowledge)).map(lambda X, y: (*X,y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            val_ds = val_ds.batch(self.args.search_batch_size).prefetch(tf.data.experimental.AUTOTUNE).with_options(options)
            self.val_ds = self.strategy.experimental_distribute_dataset(val_ds)

            (self.Knowledge, self.ori_acc, self.ori_loss, self.ori_score) = (_, self.cur_acc, self.cur_loss, self.ori_score) = self.Eval()
            self.num_t = 1            

        else:
            self.Knowledge = (self.Knowledge * self.num_t + Knowledge) / (self.num_t + 1)
            self.num_t+= 1
        
        Knowledge = tf.data.Dataset.from_tensor_slices(self.Knowledge)
        val_ds = tf.data.Dataset.zip((self.val_ds_ori, Knowledge)).map(lambda X, y: (*X,y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_ds = val_ds.batch(self.args.search_batch_size).prefetch(tf.data.experimental.AUTOTUNE).with_options(options)
        self.val_ds = self.strategy.experimental_distribute_dataset(val_ds)

def get_pruned_network(model):
    attrs = {
        'conv': ['kernel','biases'],
        'bn': ['moving_mean','moving_variance','gamma','beta']
    }
    for k, layer in model.Layers.items():
        attr = attrs['bn'] if 'bn' in k else attrs['conv']

        for a in attr:
            if not(hasattr(layer, a)):
                continue
            tensor = getattr(layer, a).numpy()
            name = getattr(layer, a).name
            trainable = getattr(layer, a).trainable

            if 'kernel' in a:
                norm = tf.linalg.norm(tensor)

            if hasattr(layer, 'in_mask') and a == 'kernel':
                tensor = tensor[..., layer.in_mask.get_mask().numpy().reshape(-1).astype(bool),:]
            if hasattr(layer, 'out_mask'):
                tensor = tensor[..., layer.out_mask.get_mask().numpy().reshape(-1).astype(bool)]

            if 'kernel' in a:
                tensor = tf.linalg.l2_normalize(tensor)*norm

            delattr(layer, a)
            setattr(layer, a, tf.Variable(tensor, trainable = trainable, name = name[:-2]))

    for k, layer in model.Layers.items():
        if hasattr(layer, 'in_mask'):
            delattr(layer, 'in_mask')
        if hasattr(layer, 'out_mask'):
            delattr(layer, 'out_mask')
    return model

def build_memory_bank(args, model, strategy, History, order_rate_var):
    History = [History[h] for h in np.argsort([l for _,l in History])]
    loss_list = np.array([l + i*1e-12 for i, (_,l) in enumerate(History)])

    min_loss = loss_list[0]
    max_loss = loss_list[-1]
    target_loss = np.arange(min_loss, max_loss, (max_loss - min_loss)/args.num_teacher).tolist()

    teachers = []
    teacher_loss = []
    for tl in target_loss:
        teachers.append(np.argmin(np.abs(tl - loss_list)))
        teacher_loss.append(loss_list[teachers[-1]])
        loss_list[teachers[-1]] = 1e6

    History = [History[t] for t in teachers]

    train_ds = CIFAR.build_memory_bank(args, model, strategy, History, order_rate_var)
    return train_ds
