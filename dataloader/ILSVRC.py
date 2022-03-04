import glob, os
import tensorflow as tf
import numpy as np

JPEG_OPT = {'fancy_upscaling': True, 'dct_method': 'INTEGER_ACCURATE'}

def build_dataset_providers(args, strategy, test_only = False, seed = 0):
    args.num_classes = 1000
    args.input_size = [224,224,3]
    args.iter_len = {}
    args.cardinality = {}

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    test_ds = ILSVRC(args, 'val', shuffle = False)
    test_ds = test_ds.map(pre_processing(is_training = False), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    args.cardinality['test'] = int(test_ds.cardinality().numpy())
    test_ds = test_ds.batch(args.val_batch_size)
    args.iter_len['test'] = int(test_ds.cardinality().numpy())
    test_ds = test_ds.with_options(options)
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

    if test_only:
        return {'test': test_ds}

def ILSVRC(args, split = 'train', sample_rate = None, shuffle = False, seed = None, sub_ds = None, saved = False):
    if split == 'val':
        image_paths = glob.glob(os.path.join(args.data_path, split, '*'))
        image_paths.sort()

        with open(os.path.join(args.home_path, 'val_gt.txt'),'r') as f:
            labels = f.readlines()

    print (split + ' dataset length :', len(image_paths))
    label_arry = np.int64(labels)
    img_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(label_arry)
    dataset = tf.data.Dataset.zip((img_ds, label_ds))

    return dataset

def resizeshortest(shape, size):
    h, w = shape[:2]
    scale = size / min(h, w)
    if h < w:
        newh, neww = size, int(scale * w + 0.5)
    else:
        newh, neww = int(scale * h + 0.5), size
    return np.int32(newh), np.int32(neww)


def pre_processing(is_training = False, contrastive = False):
    def test(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image, channels = 3, **JPEG_OPT)

        newh,neww = tf.numpy_function(resizeshortest, [tf.shape(image, tf.int32), 256], [tf.int32, tf.int32])

        image = tf.image.resize(image, (newh,neww), method='bicubic')
        image = tf.slice(image, [newh//2-112,neww//2-112,0],[224,224,-1])
        image = (image-np.array([123.675, 116.28 , 103.53 ]))/np.array([58.395, 57.12, 57.375])
        return image, label
    return test
