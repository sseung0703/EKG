import os
import tensorflow as tf
import numpy as np

def build_dataset_providers(args, strategy, test_only = False):
    if args.dataset == 'CIFAR10':
        train_images, train_labels, test_images, test_labels, pre_processing = Cifar10(args)
    if args.dataset == 'CIFAR100':
        train_images, train_labels, test_images, test_labels, pre_processing =  Cifar100(args)

    args.num_classes = int(args.dataset[5:])
    args.input_size = [32,32,3]
    args.iter_len = {}
    args.cardinality = {}

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    args.cardinality['test'] = int(test_ds.cardinality().numpy())
    test_ds = test_ds.batch(args.val_batch_size, drop_remainder=getattr(args, 'searching', False))
    test_ds = test_ds.map(pre_processing(is_training = False), num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
    test_ds = test_ds.with_options(options)
    args.iter_len['test'] = int(test_ds.cardinality().numpy())
    test_ds = strategy.experimental_distribute_dataset(test_ds.prefetch(tf.data.experimental.AUTOTUNE))

    if test_only:
        return {'test': test_ds, 'num_classes' : int(args.dataset[5:])}

    if getattr(args, 'searching', False):
        seed = 1
        def sampler(sample_rate):
            sampled_idx = []
            for l in range(np.max(train_labels)+1):
                idx = np.where(train_labels.reshape(-1) == l)[0]
                sampled_idx.append( idx[:sample_rate] if sample_rate > 0 else idx[:sample_rate-1:-1] )
            sampled_idx = np.hstack(sampled_idx)
            np.random.seed(seed)
            np.random.shuffle(sampled_idx)
            return train_images[sampled_idx], train_labels[sampled_idx]

        val_ds = tf.data.Dataset.from_tensor_slices(sampler(-32))
        args.cardinality['val'] = int(val_ds.cardinality().numpy())
        val_ds = val_ds.map(pre_processing(is_training = False), num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
        args.iter_len['val'] = args.cardinality['val']//args.search_batch_size

        train_sub_ds = tf.data.Dataset.from_tensor_slices(sampler(256))
        args.cardinality['train_sub'] = int(train_sub_ds.cardinality().numpy())
        train_sub_ds = train_sub_ds.batch(args.search_batch_size, drop_remainder=True)
        train_sub_ds = train_sub_ds.map(pre_processing(is_training = False), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        args.iter_len['train_sub'] = args.cardinality['train_sub']//args.search_batch_size
        train_sub_ds = strategy.experimental_distribute_dataset(train_sub_ds.prefetch(tf.data.experimental.AUTOTUNE))


        datasets = {
            'val': val_ds,
            'test': test_ds,
            'train_sub': train_sub_ds,
        }
        print('Datasets are built')
        return datasets

    elif args.Knowledge:
        train_ds = [train_images, train_labels, np.load(os.path.join(os.path.split(args.trained_param)[0], args.dataset + '_knowledge.npy'))]
    
    else:
        train_ds = [train_images, train_labels]
    
    train_ds = tf.data.Dataset.from_tensor_slices(tuple(train_ds)).cache()
    args.cardinality['train'] = int(train_ds.cardinality().numpy())
    train_ds = train_ds.shuffle(100*args.batch_size, seed = args.seed).batch(args.batch_size, drop_remainder = True)
    train_ds = train_ds.map(pre_processing(is_training = True),  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.with_options(options)
    args.iter_len['train'] = int(train_ds.cardinality().numpy())
    train_ds = strategy.experimental_distribute_dataset(train_ds.prefetch(tf.data.experimental.AUTOTUNE).repeat( args.train_epoch ))

    datasets = {
        'train': train_ds,
        'test': test_ds,
    }
    print('Datasets are built')
    return datasets

def Cifar10(args):
    from tensorflow.keras.datasets.cifar10 import load_data
    (train_images, train_labels), (val_images, val_labels) = load_data()
    
    def pre_processing(is_training = False):
        def training(image, *argv):
            image = tf.cast(image, tf.float32)
            B,H,W,C = tf.unstack(tf.shape(image))

            def f(image):
                B,H,W,C = tf.unstack(tf.shape(image))
                image = tf.image.random_brightness(image, .2)
                image = tf.image.random_contrast(image, .8, 1.2)
                image = tf.image.random_saturation(image, .8, 1.2)                
                image = (image-np.array([113.9,123.0,125.3]))/np.array([66.7,62.1,63.0])
                image = tf.image.random_flip_left_right(image)
                image = tf.pad(image, [[0,0],[4,4],[4,4],[0,0]], 'REFLECT')
                image = tf.image.random_crop(image, [B,H,W,C])
                return image
            
            if len(argv) == 2:
                image =  tf.stack([f(image) for _ in range(2)], 1)
            else:
                image = (image-np.array([113.9,123.0,125.3]))/np.array([66.7,62.1,63.0])
                image = tf.image.random_flip_left_right(image)
                image = tf.pad(image, [[0,0],[4,4],[4,4],[0,0]], 'REFLECT')
                image = tf.image.random_crop(image, [B,H,W,C])

            return [image] + [arg for arg in argv]
        
        def inference(image, label):
            image = tf.cast(image, tf.float32)
            image = (image-np.array([113.9,123.0,125.3]))/np.array([66.7,62.1,63.0])
            return image, label
        
        return training if is_training else inference
    return train_images, train_labels, val_images, val_labels, pre_processing

def Cifar100(args):
    from tensorflow.keras.datasets.cifar100 import load_data
    (train_images, train_labels), (val_images, val_labels) = load_data()

    def pre_processing(is_training = False):
        @tf.function
        def training(image, *argv):
            image = tf.cast(image, tf.float32)
            B,H,W,C = tf.unstack(tf.shape(image))

            def f(image):
                image = tf.image.random_brightness(image, .2)
                image = tf.image.random_contrast(image, .8, 1.2)
                image = tf.image.random_saturation(image, .8, 1.2)                
                image = (image-np.array([112,124,129]))/np.array([70,65,68])
                image = tf.image.random_flip_left_right(image)
                image = tf.pad(image, [[0,0],[4,4],[4,4],[0,0]], 'REFLECT')
                image = tf.image.random_crop(image, [B,H,W,C])
                return image
            
            if len(argv) == 2:
                image =  tf.stack([f(image) for _ in range(2)], 1)
            else:
                image = (image-np.array([112,124,129]))/np.array([70,65,68])
                image = tf.image.random_flip_left_right(image)
                image = tf.pad(image, [[0,0],[4,4],[4,4],[0,0]], 'REFLECT')
                image = tf.image.random_crop(image, [B,H,W,C])

            return [image] + [arg for arg in argv]
        
        @tf.function
        def inference(image, label):
            image = tf.cast(image, tf.float32)
            image = (image-np.array([112,124,129]))/np.array([70,65,68])
            return image, label
        
        return training if is_training else inference
    return train_images, train_labels, val_images, val_labels, pre_processing

def build_memory_bank(args, model, strategy, History, order_rate_var):
    if args.dataset == 'CIFAR10':
        train_images, train_labels, test_images, test_labels, pre_processing = Cifar10(args)
    if args.dataset == 'CIFAR100':
        train_images, train_labels, test_images, test_labels, pre_processing =  Cifar100(args)

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).cache()
    train_ds = train_ds.map(pre_processing(is_training = False),  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.batch(args.val_batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    train_ds = strategy.experimental_distribute_dataset(train_ds)

    @tf.function(experimental_compile = args.compile)
    def eval_step(images):
        return model(images, training = True)

    def eval_step_dist(images):
        pred = strategy.run(eval_step, args=(images,))
        return strategy.experimental_local_results(pred)

    @tf.function
    def get_emb(images):
        embs = []
        for o_r, _ in History:
            for h, v in zip(o_r, order_rate_var):
                v.assign(h)
            emb = tf.concat(eval_step_dist(images), 0)
            embs.append(emb)
        return tf.stack(embs, -1)

    Knowledge = []
    for images, _ in train_ds:
        Knowledge.append(get_emb(images.numpy()))
    Knowledge = np.concatenate(Knowledge, 0)

    np.save(os.path.join(args.train_path, args.dataset + '_knowledge'), Knowledge)
