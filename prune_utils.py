import time, time

import tensorflow as tf
import numpy as np
from math import ceil
from tqdm import tqdm
                                 
import utils
from dataloader import CIFAR

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction = tf.keras.losses.Reduction.SUM)
ce_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction = tf.keras.losses.Reduction.SUM)

def objective(pred, labels, K):
    loss = loss_object(labels, pred)
    score = loss + ce_object(tf.nn.softmax(K), pred)
    return loss, score

class Greedly_search_get_teacher:
    def __init__(self, args, model, strategy, datasets):
        self.args = args
        self.model = model
        self.strategy = strategy

        self.loss_accum = tf.keras.metrics.Sum(name='loss')
        self.score_accum = tf.keras.metrics.Sum(name='score')
        self.acc_accum = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

        self.width_list = get_initial_width(self.args.arch, self.model)
        self.width_list_temp = self.width_list.copy()
        (self.cur_p, self.cur_f) = (self.ori_p, self.ori_f) = utils.check_complexity(model, args)

        set_width(self.args, self.model, self.width_list)
        
        self.imp_search = Importance_search(args, model, strategy)
        self.val_ds_ori = datasets['val']

        @tf.function(jit_compile = args.compile)
        def compiled_eval_step(images, labels, K):
            pred = model(images, training = True)
            loss, score = objective(pred, labels, K)
            return pred, loss, score

        def eval_step(images, labels, K):
            pred, loss, score = compiled_eval_step(images, labels, K)
            self.loss_accum.update_state(loss)
            self.score_accum.update_state(score)
            self.acc_accum.update_state(labels, pred)
            return pred

        @tf.function
        def dist_eval_step(images, labels, K):
            return tf.concat(strategy.experimental_local_results(strategy.run(eval_step, args=(images, labels, K))), 0)

        self.eval_step = dist_eval_step

        print ('Evaluate pre-trained model and Compute knowledge')
        self.update_Knowledge()

        self.step = 0
        self.num_o = len(self.width_list)
        print ('Original Acc:', self.ori_acc, 'Original loss:', self.ori_loss)

    def Eval(self):
        Knowledge = np.concatenate([self.eval_step(images, labels, K).numpy() for i, ((images, labels), K) in enumerate(self.val_ds)], 0)
        
        acc = self.acc_accum.result().numpy()
        loss = self.loss_accum.result().numpy()/self.args.cardinality['val']
        score = self.score_accum.result().numpy()/self.args.cardinality['val']
        self.acc_accum.reset_states()
        self.loss_accum.reset_states()
        self.score_accum.reset_states()

        return Knowledge, acc, loss, score

    def update_Knowledge(self, new_Knowledge = None):
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        if new_Knowledge is None:
            # Gen dummy
            self.Knowledge = np.zeros([self.args.cardinality['val'], self.args.num_classes], dtype = np.float32) 
            Knowledge = tf.data.Dataset.from_tensor_slices(self.Knowledge).batch(self.args.val_batch_size)
            Knowledge = Knowledge.with_options(options)
            val_ds = tf.data.Dataset.zip((self.val_ds_ori, Knowledge))
            self.val_ds = self.strategy.experimental_distribute_dataset(val_ds)

            (self.Knowledge, self.ori_acc, self.ori_loss, _) = (_, self.cur_acc, self.cur_loss, _) = self.Eval()
            self.num_t = 1

        else:
            self.Knowledge = self.Knowledge*self.num_t/(self.num_t+1) + new_Knowledge/(self.num_t+1)
            self.num_t+= 1

        Knowledge = tf.data.Dataset.from_tensor_slices( self.Knowledge ).batch(self.args.val_batch_size)
        Knowledge = Knowledge.with_options(options)
        val_ds = tf.data.Dataset.zip((self.val_ds_ori, Knowledge)).prefetch(tf.data.experimental.AUTOTUNE)
        self.val_ds = self.strategy.experimental_distribute_dataset(val_ds)
        self.imp_search.val_ds = self.val_ds

    def fine_search(self, i):
        p, f = set_width(self.args, self.model, self.width_list)
        W = self.width_list[i]

        for w in np.arange(W, W + self.args.search_step, self.args.search_step *.1):
            self.width_list[i] = round(w, 2)
            p, f = set_width(self.args, self.model, self.width_list)

            if f/self.ori_f > self.args.target_rate:
                self.width_list[i] = round(w - self.args.search_step *.1, 2)
                p, f = set_width(self.args, self.model, self.width_list)
                break

        return p, f

    def check_next_step(self, i):
        self.width_list_temp[i] = self.width_list[i]

        self.width_list[i] = round(self.width_list[i] - self.search_step_list[i], 5)
        
        p, f = set_width(self.args, self.model, self.width_list)
        if f/self.ori_f < self.args.target_rate:
            p, f = self.fine_search(i)

        Knowledge, acc, loss, score = self.Eval()

        self.width_list_temp[i], self.width_list[i] = self.width_list[i], self.width_list_temp[i]
        return Knowledge, acc, loss, score, p, f

    def assign_new_width(self, losses_list, accuracy_list, importance_list, idx):
        if not(hasattr(self, 'width_history')):
            self.width_history = [[self.width_list.copy(), importance_list.copy(), self.cur_loss.copy()]]

        self.width_list[idx] = self.width_list_temp[idx]
        self.cur_p, self.cur_f = set_width(self.args, self.model, self.width_list)
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

        self.width_history.append([self.width_list.copy(), importance_list.copy(), self.cur_loss.copy()])

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

    def layer_scoring(self, scores):
        scores_ = []
        for (_,s),w,ss in zip(scores.items(), self.width_list, self.search_step_list):
            s = np.sort(s)
            if w > 0:
                w = 1. - w
                s = s[round(w * len(s)):round((w + ss) * len(s))]
                scores_.append(sum(s))
            else:
                scores_.append(1e12)
        score_th = self.trimed_otsu([s for s in scores_ if s < 1e12])
        do_check = np.array(scores_) <= score_th
        print ('# of layers to evaluated is %d/%d (%.2f)'%(np.sum(do_check), len(do_check), np.sum(do_check)/len(do_check) * 100 ))
        return do_check

    def assign_new_rates(self, scores, rates):
        step = self.args.search_step
        self.search_step_list = [ max(1/score.size, step/rate) for (_,score), rate in zip(scores.items(), rates)]
        
    def search_step(self):
        # search next steps
        scores, rates = self.imp_search.compute()
        self.assign_new_rates(scores, rates)
        do_check = self.layer_scoring(scores)

        accuracy_list = np.zeros([self.num_o])
        losses_list = np.ones([self.num_o]) * 1e12
        scores_list = np.ones([self.num_o]) * 1e12
        knowledge_list = [0]*self.num_o

        for i in tqdm(range(self.num_o)):
            if self.width_list[i] > 0 and do_check[i]:
                Knowledge, acc, loss, score, p, f = self.check_next_step(i)
                accuracy_list[i] = acc
                losses_list[i] = loss
                scores_list[i] = score
                knowledge_list[i] = Knowledge
                checked = True
        # assign the best step
        best_idx = np.argmin( scores_list )
                        
        self.assign_new_width(losses_list, accuracy_list, scores, best_idx)
        return knowledge_list[best_idx]

    def run(self):
        summary_writer = tf.summary.create_file_writer(self.args.train_path, flush_millis = 30000)
        with summary_writer.as_default():
            tf.summary.scalar('Greed_search/acc', self.cur_acc, step=self.step)
            tf.summary.scalar('Greed_search/loss', self.cur_loss, step=self.step)
            tf.summary.scalar('Greed_search/flops_rate', self.cur_f/self.ori_f, step=self.step)
            summary_writer.flush()
            
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            while (self.cur_f/self.ori_f > self.args.target_rate):
                knowledge = self.search_step()
                self.update_Knowledge(knowledge)
                summary_writer.flush()

        return self.width_history

class Importance_search:
    def __init__(self, args, model, strategy):
        model( np.zeros([1] + args.input_size, dtype = np.float32) )
        gates = []
        for k in model.Layers.keys():
            if getattr(model.Layers[k], 'get_score', False):
                gates.append(model.Layers[k].kernel)

        if args.num_classes < 1000:
            T = 4
        else:
            T = 10

        def kld(x, y, axis = -1, keepdims=True):
            return T*tf.reduce_sum(tf.nn.softmax(x/T, axis)*(tf.nn.log_softmax(x/T, axis) - tf.nn.log_softmax(y/T, axis)), axis, keepdims=keepdims)

        @tf.function(jit_compile = args.compile)
        def compiled_scoring_step(images, labels, K):
            with tf.GradientTape(watch_accessed_variables = False) as tape:
                tape.watch(gates)
                pred = model(images, training = True)
                loss, score = objective(pred, labels, K)
                loss /= args.val_batch_size
                score /= args.val_batch_size
            return tape.gradient(score, gates)

        def scoring_step(images, labels, K):
            return compiled_scoring_step(images, labels, K)

        @tf.function
        def dist_scoring_step(images, labels, K):
            gradients = strategy.run(scoring_step, args=(images, labels, K))
            gradients = [strategy.reduce(tf.distribute.ReduceOp.SUM, g, axis=None) for g in gradients]
            return gradients

        self.args = args
        self.model = model
        self.gates = gates
        self.dist_step = dist_scoring_step

    def compute(self):
        t = time.time()
        for l in self.model.Layers:
            L = self.model.Layers[l]
            if hasattr(L, 'get_score'):
                L.scoring = True

        layer_score_temp = {}
        for i, ((images, labels), K) in enumerate(self.val_ds):
            for s, v in zip(self.dist_step(images, labels, K), self.gates):
                if v.name not in layer_score_temp:
                    layer_score_temp[v.name] = s.numpy().reshape(-1)
                else:
                    layer_score_temp[v.name]+= s.numpy().reshape(-1)
        layer_score = {}
        mean_dx = {}
        norm = []
        for k in layer_score_temp:
            name = k[len(self.model.name)+1:-len('kernel:0')-1]
            layer_score[name] = 0.
            
            for in_ in self.model.in_to_out[name]:
                if in_ not in layer_score:
                    layer_score[in_] = layer_score_temp[k]
                else:
                    layer_score[in_]+= layer_score_temp[k]
 
        for l in self.model.Layers:
            L = self.model.Layers[l]
            if hasattr(L, 'get_score'):
                L.scoring = False
       
        scores_list, rate_list = layer_to_net_score(self.args, self.model, layer_score)
        assign_net_score(self.args, self.model, scores_list)
        print ('Importance searching time: %.2f'%(time.time()-t))

        return scores_list, rate_list

def build_memory_bank(args, model, strategy, History, pruned):
    History = [History[h] for h in np.argsort([l for _,_,l in History])]
    loss_list = np.array([l + i*1e-12 for i, (_, _, l) in enumerate(History)])
                
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
    train_ds = CIFAR.build_memory_bank(args, model, strategy, History)
    return train_ds

def get_initial_width(arch, model):
    if 'ResNet' in arch:
        width_list = []
        for k in model.Layers.keys():
            if 'bn' in k:
                if 'bn2' not in k:
                    width_list.append(1.)

            if ('conv' in k or 'dummy' in k):
                if getattr(model.Layers[k], 'type', 'mid') != 'input':
                    model.Layers[k].get_score = True

    return width_list

def set_slimmed_param(layer, attr, in_depth = None, out_depth = None, actual = False, trainable = False):
    if actual:
        for a in attr:
            if not(hasattr(layer, a)):
                continue
            tensor = getattr(layer, a).numpy()
            name = getattr(layer, a).name

            if 'kernel' in a:
                norm = tf.linalg.norm(tensor)

            if 'fc' in name:
                tensor = np.expand_dims(np.expand_dims(tensor,0),0)

            if in_depth is not None:
                in_depth = max(in_depth, 1/tensor.shape[-2])
                Di = ceil(tensor.shape[-2]*in_depth)
                tensor = tensor[:,:,layer.in_mask.numpy() < Di]
                
            if out_depth is not None:
                out_depth = max(out_depth, 1/tensor.shape[-1])
                Do = ceil(tensor.shape[-1]*out_depth)
                tensor = tensor[:,:,:,layer.out_mask.numpy() < Do]

            if 'fc' in name:
                tensor = tensor.reshape(*tensor.shape[2:])

            if 'kernel' in a:
                tensor = tf.linalg.l2_normalize(tensor)*norm

            delattr(layer, a)
            setattr(layer, a, tf.Variable(tensor, trainable = trainable, name = name[:-2]))

    else:
        tensor = getattr(layer, attr[0])
        name = getattr(layer, attr[0]).name

        if in_depth is not None:
            Di = tensor.shape[-2]
            in_depth = max(in_depth, 1/Di)
            if not(hasattr(layer, 'in_mask')):
                layer.in_mask = layer.add_weight(name  = layer.name +'/in_mask', 
                                      shape = [Di],
                                      initializer = tf.constant_initializer(np.arange(Di).astype(np.float32)),
                                      trainable = False)
            if not(hasattr(layer, 'in_depth')):
                layer.in_depth = layer.add_weight(name  = layer.name +'/in_depth', 
                                      shape = [],
                                      initializer = tf.constant_initializer(1.),
                                      trainable = False)
            else:
                layer.in_depth.assign(in_depth)

        if out_depth is not None:
            Do = tensor.shape[-1]
            out_depth = max(out_depth, 1/Do)

            if not(hasattr(layer, 'out_mask')):
                layer.out_mask = tf.Variable(np.arange(Do).astype(np.float32), trainable = False, name = layer.name + '/out_mask')
            if not(hasattr(layer, 'out_depth')):
                layer.out_depth = tf.Variable(1., trainable = False, name = layer.name + '/out_depth')
            else:
                layer.out_depth.assign(out_depth)

def set_width(args, model, width_list, actual = False, check_complex = True):
    if 'ResNet' in args.arch:
        w_num = 0
        for k in model.Layers.keys():
            layer = model.Layers[k]
            if 'conv' in k:
                if 'conv2' in k or 'conv3' in k:
                    if 'conv3' in k:
                        group_width = width_list[w_num]
                        w_num += 1
                    
                    Di = in_width
                    Do = group_width

                    if 'conv2' in k:
                        in_width = group_width
                
                else:
                    width = width_list[w_num]
                    if layer.type == 'input':
                        group_width = width
                        Di = None
                    else:
                        Di = in_width
                    Do = width
                    in_width = width
                    w_num += 1
                
                set_slimmed_param(layer, ['kernel', 'bias'], in_depth = Di, out_depth = Do, actual = actual, trainable = True)

                if k.replace('conv', 'bn') in model.Layers:
                    bn = model.Layers[k.replace('conv', 'bn')]
                    set_slimmed_param(bn, ['moving_mean', 'moving_variance'], out_depth = Do, actual = actual, trainable = False)
                    set_slimmed_param(bn, ['gamma', 'beta'], out_depth = Do, actual = actual, trainable = True)

            if 'fc' in k:
                set_slimmed_param(layer, ['kernel', 'bias'], in_depth = Do, actual = actual, trainable = True)

    if actual:
        clear_width(model)

    if check_complex:
        return utils.check_complexity(model, args)

def clear_width(model):
    for k in model.Layers.keys():
        if hasattr(model.Layers[k], 'in_depth'):
            delattr(model.Layers[k],'in_depth')
            delattr(model.Layers[k],'in_mask')
        if hasattr(model.Layers[k], 'out_depth'):
            delattr(model.Layers[k],'out_depth')
            delattr(model.Layers[k],'out_mask')
        
def layer_to_net_score(args, model, layer_score):
    if 'ResNet' in args.arch:
        scores = {}
        rate_list = []
        w_num = 0
        for k in model.Layers.keys():
            layer = model.Layers[k]
            if 'conv' in k:
                s = layer_score[k]

                if 'conv3' in k or layer.type == 'input':
                    group_name = k
                    group_idx = w_num
                    scores[group_name] = s
                    rate_list.append(1)
                    w_num += 1

                elif 'conv2' in k:
                    scores[group_name] += s
                    rate_list[group_idx] += 1

                else:
                    scores[k] = s
                    rate_list.append(1)
                    w_num += 1
    return scores, rate_list

def assign_net_score(args, model, net_score):
    if 'ResNet' in args.arch:    
        for k in model.Layers.keys():
            layer = model.Layers[k]
            if 'conv' in k:
                if 'conv3' in k:
                    group_s = net_score[k]
                    imp = tf.constant(utils.sorted_idx(group_s, reverse = True).astype(np.float32))
                    group_imp = imp
                    layer.in_mask.assign(prev_imp)
                    layer.out_mask.assign(group_imp)

                elif 'conv2' in k:
                    layer.in_mask.assign(prev_imp)
                    layer.out_mask.assign(group_imp)
                    prev_imp = group_imp
                else:
                    s = net_score[k]
                    imp = tf.constant(utils.sorted_idx(s, reverse = True).astype(np.float32))
                    if layer.type == 'input':
                        group_s = s
                        group_imp = imp
                        layer.out_mask.assign(imp)
                    else:
                        layer.in_mask.assign(prev_imp)
                        layer.out_mask.assign(imp)

                    prev_imp = imp

                if k.replace('conv', 'bn') in model.Layers:
                    bn = model.Layers[k.replace('conv', 'bn')]
                    bn.out_mask.assign(layer.out_mask)

            if 'fc' in k:
                layer.in_mask.assign(prev_imp)

