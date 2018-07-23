import keras
import keras.backend as K
from keras_model2 import PixelLink
from keras_model2 import get_loss
import config
import pixel_link
from preprocessing import ssd_vgg_preprocessing
import tensorflow as tf
import sys
import numpy as np
from scipy import misc
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


def get_data_slim():
    import datasets.dataset_factory as dataset_factory

    config_initialization()
    dataset = dataset_factory.get_dataset(dataset_name=config.dataset_name, split_name='train',
                                          dataset_dir=config.dataset_path)
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


def generate_random_data():
    import numpy as np
    height = config.train_image_height
    width = config.train_image_width
    size = 2
    y_true = np.random.random_sample([size, height // config.strides[0], width // config.strides[0], 18])
    # y_true = np.random.random_sample([size, height // config.strides[0], width // config.strides[0], 1])
    b_image = np.random.random_sample([size, height, width, 3])
    return b_image, y_true


# generate batch data
def data_generator(imgs, labels, batch_size):


    while True:
        index = np.random.choice(len(imgs), batch_size)
        batch_labels = labels[index]
        batch_imgs = imgs[index, :, :, :]

        yield batch_imgs, batch_labels


def get_data():
    from pathlib import Path
    import json
    import os
    data_dir = config.train_dir
    data_path = Path(data_dir)
    labels_path = os.path.join(config.train_labels_path, config.train_labels_name)

    with open(labels_path) as f:
        labels = json.load(f)

    label_list = []
    img_list = []
    for image_name in sorted(data_path.glob("*.jpg")):
        img = misc.imread(str(image_name))
        img = misc.imresize(img, [config.train_image_height, config.train_image_width, 3])
        img_name = str(image_name).split('/')[-1]
        label = labels[img_name]
        gxs = label['gxs']
        gys = label['gys']
        glabel = label['glabel']
        pixel_cls_label, pixel_cls_weight, pixel_link_label, pixel_link_weight = \
            pixel_link.cal_gt_for_single_image(gxs, gys, glabel)

        pixel_cls_label.astype(np.float32)
        stack = np.stack([pixel_cls_label, pixel_cls_weight], axis=2)

        pixel_link_label.astype(np.float32)
        y_true = np.concatenate([stack, pixel_link_label, pixel_link_weight], axis=-1)

        label_list.append(y_true)
        img_list.append(img)

    img_list = np.array(img_list)
    label_list = np.array(label_list)
    print(img_list.shape, label_list.shape)
    return img_list, label_list



# def get_data():
#     [image, glabel, gbboxes, x1, x2, x3, x4, y1, y2, y3, y4] = get_data_slim()  # this get a single image
#     gxs = K.transpose(K.stack([x1, x2, x3, x4]))  # shape = (N, 4) N is number of bboxes
#     gys = K.transpose(K.stack([y1, y2, y3, y4]))
#
#     image, glabel, gbboxes, gxs, gys = \
#         ssd_vgg_preprocessing.preprocess_image(
#             image, glabel, gbboxes, gxs, gys,
#             out_shape=config.train_image_shape,
#             data_format=config.data_format,
#             use_rotation=config.use_rotation,
#             is_training=True)
#     return [image, glabel, gbboxes, gxs, gys]



def train_model():

    # [image, glabel, gbboxes, gxs, gys] = get_data()
    #
    # pixel_cls_label, pixel_cls_weight, \
    # pixel_link_label, pixel_link_weight = \
    #     pixel_link.tf_cal_gt_for_single_image(gxs, gys, glabel)

    # b_image, b_pixel_cls_label, b_pixel_cls_weight, b_pixel_link_label, b_pixel_link_weight = \
    #     K.tf.train.shuffle_batch([image, pixel_cls_label, pixel_cls_weight,
    #                               pixel_link_label, pixel_link_weight],
    #                              batch_size=config.batch_size,
    #                              capacity=200,
    #                              min_after_dequeue=100,
    #                              num_threads=32)

    image_set, true_set = get_data()
    # print(type(image_set[0]), type(true_set[0]))


    # batch_queue = slim.prefetch_queue.prefetch_queue(
    #     [b_image, b_pixel_cls_label, b_pixel_cls_weight,
    #      b_pixel_link_label, b_pixel_link_weight],
    #     capacity=50)
    #
    # b_image, b_pixel_cls_label, b_pixel_cls_weight, b_pixel_link_label, b_pixel_link_weight = \
    #     batch_queue.dequeue()
    pl_net = PixelLink()

    # compile our own loss function
    pl_net.model.compile(loss=get_loss, optimizer='sgd', metrics=['accuracy'])
    pl_net.model.summary()

    # pl_net.model.compile(loss=losses.sparse_categorical_crossentropy, optimizer='sgd', metrics=['accuracy'])
    # b_image, y_true = generate_random_data()
    print('start training')
    # pl_net.model.train_on_batch(b_image, y_true)

    train_generator = data_generator(image_set, true_set, config.batch_size)
    pl_net.model.fit_generator(train_generator, epochs=50, steps_per_epoch=60)
    pl_net.model.save()


train_model()
