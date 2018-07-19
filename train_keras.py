import keras
import keras.backend as K
from keras_model2 import PixelLink
from keras_model2 import get_loss
import config
import pixel_link
from preprocessing import ssd_vgg_preprocessing
import tensorflow as tf
import sys


sys.path.append('/Users/ci.chen/src/pixel_link_mobile/pylib/src')
import util
import tensorflow.contrib.slim as slim


def config_initialization():
    # image shape and feature layers shape inference
    config.default_config()
    image_shape = (config.train_image_height, config.train_image_width)

    if not config.dataset_path:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.DEBUG)
    util.init_logger(
        log_file='log_train_pixel_link_%d_%d.log' % image_shape,
        log_path=config.train_dir, stdout=False, mode='a')

    # config.load_config(config.train_dir)
    config.init_config(image_shape,
                       batch_size=config.batch_size,
                       weight_decay=config.weight_decay,
                       num_gpus=config.num_gpus
                       )
    config.default_config()
    config.score_map_shape = (config.train_image_height // config.strides[0],
                              config.train_image_width // config.strides[0])
    height = config.train_image_height
    score_map = config.score_map_shape
    stride = config.strides[0]
    batch_size = config.batch_size
    batch_size_per_gpu = config.batch_size_per_gpu

    util.proc.set_proc_name('train_pixel_link_on' + '_' + config.dataset_name)


def get_data():
    data_path = config.dataset_path + config.train_name

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/x1': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/x2': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/x3': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/x4': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/y1': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/y2': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/y3': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/y4': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
    }
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_data = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_data, features=keys_to_features)

    return []


def get_data_slim():
    import datasets.dataset_factory as dataset_factory

    config_initialization()
    dataset = dataset_factory.get_dataset(dataset_name=config.dataset_name, split_name='train',
                                          dataset_dir='/Users/ci.chen/Downloads/ic15/pixel_link/Challenge4/')
    with tf.name_scope(config.dataset_name + '_data_provider'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=config.num_readers,
            common_queue_capacity=1000 * config.batch_size,
            common_queue_min=700 * config.batch_size,
            shuffle=True)
        # Get for SSD network: image, labels, bboxes.
    [image, glabel, gbboxes, x1, x2, x3, x4, y1, y2, y3, y4] = provider.get([
        'image',
        'object/label',
        'object/bbox',
        'object/oriented_bbox/x1',
        'object/oriented_bbox/x2',
        'object/oriented_bbox/x3',
        'object/oriented_bbox/x4',
        'object/oriented_bbox/y1',
        'object/oriented_bbox/y2',
        'object/oriented_bbox/y3',
        'object/oriented_bbox/y4'
    ])

    return [image, glabel, gbboxes, x1, x2, x3, x4, y1, y2, y3, y4]


def train_model():
    [image, glabel, gbboxes, x1, x2, x3, x4, y1, y2, y3, y4] = get_data_slim()
    gxs = K.transpose(K.stack([x1, x2, x3, x4]))  # shape = (N, 4)
    gys = K.transpose(K.stack([y1, y2, y3, y4]))

    image, glabel, gbboxes, gxs, gys = \
        ssd_vgg_preprocessing.preprocess_image(
            image, glabel, gbboxes, gxs, gys,
            out_shape=config.train_image_shape,
            data_format=config.data_format,
            use_rotation=config.use_rotation,
            is_training=True)

    pixel_cls_label, pixel_cls_weight, \
    pixel_link_label, pixel_link_weight = \
        pixel_link.tf_cal_gt_for_single_image(gxs, gys, glabel)

    # b_image, b_pixel_cls_label, b_pixel_cls_weight, b_pixel_link_label, b_pixel_link_weight = \
    #     K.tf.train.shuffle_batch([image, pixel_cls_label, pixel_cls_weight,
    #                               pixel_link_label, pixel_link_weight],
    #                              batch_size=config.batch_size,
    #                              capacity=200,
    #                              min_after_dequeue=100,
    #                              num_threads=32)

    b_image, b_pixel_cls_label, b_pixel_cls_weight, \
    b_pixel_link_label, b_pixel_link_weight = \
        tf.train.batch(
            [image, pixel_cls_label, pixel_cls_weight,
             pixel_link_label, pixel_link_weight],
            batch_size=int(config.batch_size_per_gpu),
            num_threads=int(config.num_preprocessing_threads),
            capacity=500)

    batch_queue = slim.prefetch_queue.prefetch_queue(
        [b_image, b_pixel_cls_label, b_pixel_cls_weight,
         b_pixel_link_label, b_pixel_link_weight],
        capacity=50)
    b_image, b_pixel_cls_label, b_pixel_cls_weight, b_pixel_link_label, b_pixel_link_weight = \
        batch_queue.dequeue()
    pl_net = PixelLink()
    # pl_net.build_loss(pixel_cls_labels=b_pixel_cls_label,
    #                   pixel_cls_weights=b_pixel_cls_weight,
    #                   pixel_link_labels=b_pixel_link_label,
    #                   pixel_link_weights=b_pixel_link_weight)
    shape = tf.shape(b_pixel_cls_label)
    b_pixel_cls_label = tf.cast(b_pixel_cls_label, tf.float32)
    stack = tf.stack([b_pixel_cls_label, b_pixel_cls_weight], axis=3)
    # b_pixel_cls_label = tf.cast(tf.reshape(b_pixel_cls_label, [shape[0], shape[1], shape[2], 1]), tf.float32)
    # b_pixel_cls_weight = tf.reshape(b_pixel_cls_weight, [shape[0], shape[1], shape[2], 1])

    b_pixel_link_label = tf.cast(b_pixel_link_label, tf.float32)
    y_true = keras.layers.merge.concatenate \
        ([stack, b_pixel_link_label, b_pixel_link_weight], axis=-1)
    # y_true = tf.concat\
    #     ([b_pixel_cls_label, b_pixel_cls_weight, b_pixel_link_label, b_pixel_link_weight], axis=-1)

    pl_net.model.compile(loss=get_loss, optimizer='adam', metrics=['accuracy'])
    print('start training')
    pl_net.model.train_on_batch(b_image, y_true)


train_model()
