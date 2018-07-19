import tensorflow as tf
import numpy as np
import cv2

from nets import pixel_link_symbol
import pylib.src.util as util
from preprocessing import ssd_vgg_preprocessing
import config


slim = tf.contrib.slim
tf.app.flags.DEFINE_string('checkpoint_path', None,
                           'the path of pretrained model to be used. If there are checkpoints\
                            in train_dir, this config will be ignored.')

tf.app.flags.DEFINE_float('gpu_memory_fraction', -1,
                          'the gpu memory fraction to be used. If less than 0, allow_growth = True is used.')

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #


tf.app.flags.DEFINE_integer('eval_image_width', 640, 'resized image width for inference')
tf.app.flags.DEFINE_integer('eval_image_height', 360, 'resized image height for inference')
tf.app.flags.DEFINE_float('pixel_conf_threshold', None, 'threshold on the pixel confidence')
tf.app.flags.DEFINE_float('link_conf_threshold', None, 'threshold on the link confidence')

tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate.')
tf.app.flags.DEFINE_float('momentum', 0.9, 'The momentum for the MomentumOptimizer')
tf.app.flags.DEFINE_float('weight_decay', 0.0001, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_bool('using_moving_average', True, 'Whether to use ExponentionalMovingAverage')
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, 'The decay rate of ExponentionalMovingAverage')

# =========================================================================== #
tf.app.flags.DEFINE_string(
    'dataset_name', None, 'The name of the dataset to load.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', './', 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer('train_image_width', 640, 'Train image size')
tf.app.flags.DEFINE_integer('train_image_height', 360, 'Train image size')
FLAGS = tf.app.flags.FLAGS


def config_initialization():
    # image shape and feature layers shape inference
    image_shape = (FLAGS.eval_image_height, FLAGS.eval_image_width)

    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.DEBUG)

    config.init_config(image_shape,
                       batch_size=1,
                       pixel_conf_threshold=FLAGS.pixel_conf_threshold,
                       link_conf_threshold=FLAGS.link_conf_threshold,
                       num_gpus=1,
                       )


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def create_model():
    output_graph = 'frozen_model.pb'
    config_initialization()
    global_step = slim.get_or_create_global_step()
    with tf.name_scope('evaluation_%dx%d' % (FLAGS.eval_image_height, FLAGS.eval_image_width)):
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            image = tf.placeholder(dtype=tf.int32, shape=[None, None, 3], name='net/input_images')
            image_shape = tf.placeholder(dtype=tf.int32, shape=[3, ])
            processed_image, _, _, _, _ = ssd_vgg_preprocessing.preprocess_image(image, None, None, None, None,
                                                                                 out_shape=[360, 640],
                                                                                 data_format=config.data_format,
                                                                                 is_training=False)
            b_image = tf.expand_dims(processed_image, axis=0)

            # build model and loss
            net = pixel_link_symbol.PixelLinkNet(b_image, is_training=False)
            # b_image, b_pixel_cls_label, b_pixel_cls_weight, \
            # b_pixel_link_label, b_pixel_link_weight = batch_queue.dequeue()
            # net.build_loss(
            #     pixel_cls_labels,
            #     pixel_cls_weights,
            #     pixel_link_labels,
            #     pixel_link_weights,
            #     do_summary)

    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay)
    variables_to_restore = variable_averages.variables_to_restore(
        tf.trainable_variables())
    variables_to_restore[global_step.op.name] = global_step
    # open sess and then save the model.

    saver = tf.train.Saver()
    total_parameters = 0

    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        print("name:", variable.name)
        print("shape:", shape)

        for dim in shape:
            variable_parameters *= dim.value
        print('variable:', variable_parameters)
        total_parameters += variable_parameters
    print('total:', total_parameters)

    # builder = tf.saved_model.builder.SavedModelBuilder('test/')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # for op in sess.graph.get_operations():
        #     print(op.name)

        print(tf.contrib.framework.get_variables_to_restore())
        # builder.add_meta_graph_and_variables(sess,
        #                                      [tag_constants.TRAINING],
        #                                      signature_def_map=None,
        #                                      assets_collection=None)
        # tf.train.write_graph(sess.graph_def, './save/', 'mobile_net_0.01.pbtxt')
        saver.save(sess, 'save/mobile_net_0.01.ckpt')
        tf.train.write_graph(sess.graph_def, '.', 'mobile_net_0.01' + '.pb', as_text=False)
        graph_def = sess.graph_def

        from tensorflow.python.platform import gfile
        with gfile.GFile('./save/mobile_net_0.01.pbtxt', 'wb') as f:
            f.write(graph_def.SerializeToString())
    # builder.save()
        # firstly inspect the names?

        # output_graph_def = tf.graph_util.convert_variables_to_constants(
        #     sess,  # The session is used to retrieve the weights
        #     tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
        #     # The output node names are used to select the useful nodes
        #     output_node_names=output_node_names
        # )
        #
        # # Finally we serialize and dump the output graph to the filesystem
        # with tf.gfile.GFile(output_graph, "wb") as f:
        #     f.write(output_graph_def.SerializeToString())
        # print("%d ops in the final graph." % len(output_graph_def.node))


def save_model():
    from tensorflow.python.tools import freeze_graph
    from tensorflow.python.tools import optimize_for_inference_lib

    MODEL_NAME = 'mobile_net_0.01'

    input_graph_path = MODEL_NAME+'.pbtxt'
    checkpoint_path = './save/'+MODEL_NAME+'.ckpt'
    input_saver_def_path = ""
    input_binary = False
    output_node_names = "pixel_link/score_from_evaluation_360x640/MobileNet/conv_1/biases"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_frozen_graph_name = 'save/frozen_'+MODEL_NAME+'.pb'
    output_optimized_graph_name = 'optimized_'+MODEL_NAME+'.pb'
    clear_devices = True

    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_frozen_graph_name, clear_devices, "")

    # input_graph_def = tf.GraphDef()
    # with tf.gfile.Open(output_frozen_graph_name, "r") as f:
    #     data = f.read()
    #     input_graph_def.ParseFromString(data)

    # output_graph_def = optimize_for_inference_lib.optimize_for_inference(
    #     input_graph_def,
    #     ["I"],  # an array of the input node(s)
    #     ["O"],  # an array of output nodes
    #     tf.float32.as_datatype_enum)

    # Save the optimized graph

    # f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
    # f.write(output_graph_def.SerializeToString())

create_model()
save_model()



# Optimize for inference

