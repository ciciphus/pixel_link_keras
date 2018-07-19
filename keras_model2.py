import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, SeparableConv2D, BatchNormalization, AveragePooling2D
from keras.layers import Softmax, UpSampling2D, Input, concatenate, merge, Permute
from keras.layers.merge import add, concatenate
from keras.losses import sparse_categorical_crossentropy
import tensorflow as tf
import config


class PixelLink:
    def __init__(self):
        config.default_config()
        self.height = config.train_image_height
        self.width = config.train_image_width

        self.input_shape = (self.height, self.width, 3)
        # self.input = Input(tensor=image)
        self.input = Input(shape=self.input_shape)
        self.net = self.input
        self.width_multiplier = 1
        # trainign will change
        self.is_training = True
        pixel_cls_logits, pixel_link_logits = self.create_model()
        self.pixel_link_logits = pixel_link_logits
        self.pixel_cls_logits = pixel_cls_logits
        output = [pixel_cls_logits, pixel_link_logits]
        # self.model = keras.models.Model(inputs=self.input, outputs=[pixel_cls_logits, pixel_link_logits])
        merged = concatenate([pixel_cls_logits, pixel_link_logits], axis=-1)
        self.model = keras.models.Model(inputs=self.input, outputs=merged)

    # create pixel link model
    def create_model(self):
        self.net = Conv2D(filters=round(self.width_multiplier * 32), kernel_size=[3, 3],
                          input_shape=self.input_shape, padding="same")(self.net)
        self.net = self._add_df_layer(self.net, 64)
        self.net = self._add_df_layer(self.net, 128, downsample=True)
        self.net = self._add_df_layer(self.net, 128)

        num_neighbours = config.num_neighbours * 2
        # two different net, so there are two different convolution layer
        shortcut1 = self._shortcut(self.net)
        shortcut_link1 = self._shortcut(self.net, num_neighbours)

        self.net = self._add_df_layer(self.net, 256, downsample=True)
        self.net = self._add_df_layer(self.net, 256)

        shortcut2 = self._shortcut(self.net)
        shortcut_link2 = self._shortcut(self.net, num_neighbours)

        self.net = self._add_df_layer(self.net, 512, downsample=True)

        self.net = self._add_df_layer(self.net, 512)
        self.net = self._add_df_layer(self.net, 512)
        self.net = self._add_df_layer(self.net, 512)
        self.net = self._add_df_layer(self.net, 512)
        self.net = self._add_df_layer(self.net, 512)

        shortcut3 = self._shortcut(self.net)
        shortcut_link3 = self._shortcut(self.net, num_neighbours)

        self.net = self._add_df_layer(self.net, 1024, downsample=True)
        self.net = self._add_df_layer(self.net, 1024)

        # the net decide text or not
        text_net = self._shortcut(self.net)
        text_net = self._up_sample(text_net, shortcut3)
        text_net = self._up_sample(text_net, shortcut2)
        text_net = self._up_sample(text_net, shortcut1)

        # the net decide pixel linked or not
        pixel_net = self._shortcut(self.net, num_neighbours)
        pixel_net = self._up_sample(pixel_net, shortcut_link3)
        pixel_net = self._up_sample(pixel_net, shortcut_link2)
        pixel_net = self._up_sample(pixel_net, shortcut_link1)

        # last three layers of mobilenet, we may neglect this stuff
        # self.net.add(AveragePooling2D(pool_size=[7, 7]))
        # self.net.add(Dense(num_class))
        # self.net.add(Softmax())

        return text_net, pixel_net

    # def _score_layer(self, net, num_classes):
    #     net = Conv2D(filters=num_classes, kernel_size=[1, 1], strides=1)(net)
    #     use_dropout = config.dropout_ratio > 0
    #
    #     if use_dropout:
    #         if self.is_training:
    #             dropout_ratio = config.dropout_ratio
    #         else:
    #             dropout_ratio = 0
    #     keep_prob = 1.0 - dropout_ratio
    #     tf.logging.info('Using Dropout, with keep_prob = %f' % (keep_prob))
    #     net = Dropout(keep_prob=keep_prob)(net)
    #     return net

    # upsample the map and add it together
    def _up_sample(self, net, shortcut):
        net = UpSampling2D(size=(2, 2))(net)
        # net = add([net, shortcut])
        net = keras.layers.Add()([net, shortcut])
        return net

    # score layer, create a shortcut from prev conv layer
    def _shortcut(self, _input, num_pwc_filter=config.num_classes):
        res = Conv2D(filters=num_pwc_filter, kernel_size=[1, 1], padding="same")(_input)
        use_dropout = config.dropout_ratio > 0
        dropout_ratio = config.dropout_ratio
        # if use_dropout:
        #     if self.is_training:
        #         dropout_ratio = config.dropout_ratio
        #     else:
        #         dropout_ratio = 0
        # else:
        #     dropout_ratio = 0

        keep_prob = 1.0 - dropout_ratio
        tf.logging.info('Using Dropout, with keep_prob = %f' % (keep_prob))
        res = Dropout(rate=dropout_ratio)(res)

        return res

    # a unit for deepwise layer
    def _add_df_layer(self, net, num_pwc_filters, downsample=False):
        num_pwc_filters = round(num_pwc_filters * self.width_multiplier)
        _stride = 2 if downsample else 1

        net = SeparableConv2D(filters=num_pwc_filters, kernel_size=[3, 3],
                              strides=_stride, padding="same")(net)
        net = BatchNormalization()(net)
        net = Conv2D(kernel_size=[1, 1], filters=num_pwc_filters, padding="same")(net)
        net = BatchNormalization()(net)

        return net


def _flat_pixel_cls_values(values):
    shape = tf.shape(values)

    # values = keras.layers.Reshape([config.train_image_height * config.train_image_width //
    #                                (config.strides[0] * config.strides[0]), -1])(values)
    # values = tf.reshape(values, shape=[tf.shape(values)[0], -1, tf.shape(values)[-1]])

    values = tf.reshape(values, shape=[shape[0], -1, shape[-1]])
    return values


def _logits_to_scores(pixel_cls_logits, pixel_link_logits):
    pixel_cls_scores = Softmax()(pixel_cls_logits)

    pixel_cls_logits_flatten = \
        _flat_pixel_cls_values(pixel_cls_logits)
    pixel_cls_scores_flatten = \
        _flat_pixel_cls_values(pixel_cls_scores)

    #         shape = self.pixel_link_logits.shape.as_list()
    shape = tf.shape(pixel_link_logits)
    # print(tf.shape(self.pixel_link_logits))
    # reshape logits to get the 8*2 channel
    height = config.train_image_height // config.strides[0]
    width = config.train_image_width // config.strides[0]
    # pixel_link_logits = keras.layers.Reshape((width, height, config.num_neighbours, 2)) \
    #     (pixel_link_logits)
    pixel_link_logits = tf.reshape(pixel_link_logits,
                                   [shape[0], shape[1], shape[2], config.num_neighbours, 2])

    pixel_link_scores = Softmax()(pixel_link_logits)

    # pixel_pos_scores = pixel_cls_scores[:, :, :, 1]
    # link_pos_scores = pixel_link_scores[:, :, :, :, 1]

    return pixel_cls_logits_flatten, pixel_cls_scores_flatten


# custom loss function
def get_loss(y_true, y_pred):
    pixel_cls_labels = tf.cast(y_true[:, :, :], tf.int32)
    pixel_cls_weights = y_true[1]
    pixel_link_labels = tf.cast(y_true[2], tf.int32)
    pixel_link_weights = y_true[3]
    # pixel_cls_labels, pixel_cls_weights, pixel_link_labels, pixel_link_weights = y_true
    pixel_cls_logits = y_pred[:, :, :, 0:2]
    pixel_link_logits = y_pred[:, :, :, 2:18]
    pixel_cls_logits_flatten, pixel_cls_scores_flatten = _logits_to_scores(pixel_cls_logits, pixel_link_logits)


    loss = build_loss(pixel_cls_labels, pixel_cls_weights, pixel_link_labels, pixel_link_weights,
                      pixel_cls_logits_flatten, pixel_cls_scores_flatten, pixel_link_logits)
    return loss


def build_loss(pixel_cls_labels, pixel_cls_weights, pixel_link_labels, pixel_link_weights,
               pixel_cls_logits_flatten, pixel_cls_scores_flatten, pixel_link_logits):
    pixel_link_neg_loss_weight_lambda = config.pixel_link_neg_loss_weight_lambda
    pixel_cls_loss_weight_lambda = config.pixel_cls_loss_weight_lambda
    pixel_link_loss_weight = config.pixel_link_loss_weight
    batch_size = config.batch_size_per_gpu
    background_label = config.background_label
    text_label = config.text_label

    # self.pixel_cls_logits = Permute(axis=(2, 3, 1), )(self.pixel_cls_logits)
    # self.pixel_link_logits = Permute(axis=(2, 3, 1), )(self.pixel_link_logits)

    def OHNM_single_image(scores, n_pos, neg_mask):
        """Online Hard Negative Mining.
            scores: the scores of being predicted as negative cls
            n_pos: the number of positive samples
            neg_mask: mask of negative samples
            Return:
                the mask of selected negative samples.
                if n_pos == 0, top 10000 negative samples will be selected.
        """

        def has_pos():
            return n_pos * config.max_neg_pos_ratio

        def no_pos():
            return tf.constant(10000, dtype='int32')

        # n_neg = 10000 or n_pos * max_neg_pos_ratio
        # it construct a limitation of neg/pos

        n_neg = tf.cond(n_pos > 0, has_pos, no_pos)

        # max_neg entries would be no more than neg_mask num
        max_neg_entries = tf.reduce_sum(K.cast(neg_mask, dtype='int32'))

        n_neg = K.minimum(n_neg, max_neg_entries)
        n_neg = K.cast(n_neg, 'int32')

        n_neg = tf.cond(n_pos > 0, has_pos, no_pos)

        # max_neg entries would be no more than neg_mask num
        max_neg_entries = tf.reduce_sum(tf.cast(neg_mask, dtype='int32'))

        n_neg = tf.minimum(n_neg, max_neg_entries)
        n_neg = tf.cast(n_neg, dtype='int32')

        # return the negative mask
        def has_neg():
            # TODO: change tf to keras?
            neg_conf = tf.boolean_mask(scores, neg_mask)  # K.boolean_mask(scores, neg_mask)
            vals, _ = tf.nn.top_k(-neg_conf, k=n_neg)
            threshold = vals[-1]  # a negtive value
            selected_neg_mask = tf.logical_and(neg_mask, scores <= -threshold)

            return selected_neg_mask

        def no_neg():
            selected_neg_mask = tf.zeros_like(neg_mask)
            return selected_neg_mask

        # return the negative mask
        selected_neg_mask = tf.cond(n_neg > 0, has_neg, no_neg)
        return tf.cast(selected_neg_mask, dtype='int32')

        # batch version, call OHNM_image

    def OHNM_batch(neg_conf, pos_mask, neg_mask):
        selected_neg_mask = []
        for image_idx in range(batch_size):
            image_neg_conf = neg_conf[image_idx, :]
            image_neg_mask = neg_mask[image_idx, :]
            image_pos_mask = pos_mask[image_idx, :]
            n_pos = tf.reduce_sum(tf.cast(image_pos_mask, dtype='int32'))
            selected_neg_mask.append(OHNM_single_image(image_neg_conf, n_pos, image_neg_mask))

        selected_neg_mask = tf.stack(selected_neg_mask)
        return selected_neg_mask

    batch_size = int(batch_size)
    pos_pixel_weights_flatten = tf.reshape(pixel_cls_weights, [batch_size, -1])
    pixel_cls_labels_flatten = tf.reshape(pixel_cls_labels, [batch_size, -1])
    pos_mask = tf.equal(pixel_cls_labels_flatten, text_label)
    neg_mask = tf.equal(pixel_cls_labels_flatten, background_label)
    pos_cast = tf.cast(pos_mask, dtype=tf.float32)
    n_pos = tf.reduce_sum(pos_cast)

    # OHNM on pixel classification task

    with K.name_scope('pixel_cls_loss'):
        def no_pos():
            return tf.constant(.0)

        def has_pos():
            pixel_cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=pixel_cls_logits_flatten,
                labels=tf.cast(pos_mask, dtype=tf.int32))

            # logits_cast = K.cast(self.pixel_cls_logits_flatten, dtype='int32')
            # pixel_cls_loss = K.sparse_categorical_crossentropy(logits_cast, pos_mask, from_logits=True)

            pixel_neg_scores = pixel_cls_scores_flatten[:, :, 0]
            selected_neg_pixel_mask = OHNM_batch(pixel_neg_scores, pos_mask, neg_mask)

            pixel_cls_weights = pos_pixel_weights_flatten + \
                                tf.cast(selected_neg_pixel_mask, tf.float32)
            n_neg = tf.cast(tf.reduce_sum(selected_neg_pixel_mask), tf.float32)
            loss = tf.reduce_sum(pixel_cls_loss * pixel_cls_weights) / (n_neg + n_pos)
            return loss

        pixel_cls_loss = has_pos()

    with K.name_scope('pixel_link_loss'):
        def no_pos():
            return tf.constant(.0), tf.constant(.0)

        def has_pos():
            pixel_link_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=pixel_link_logits,
                labels=pixel_link_labels)

            # pixel_link_loss = K.sparse_categorical_crossentropy(self.pixel_link_logits, pixel_link_labels,
            #                                                     from_logits=True)

            def get_loss(label):
                link_mask = tf.equal(pixel_link_labels, label)
                link_weights = pixel_link_weights * tf.cast(link_mask, tf.float32)
                n_links = tf.reduce_sum(link_weights)
                loss = tf.reduce_sum(pixel_link_loss * link_weights) / n_links
                return loss

            neg_loss = get_loss(0)
            pos_loss = get_loss(1)
            return neg_loss, pos_loss

        pixel_neg_link_loss, pixel_pos_link_loss = \
            tf.cond(n_pos > 0, has_pos, no_pos)

        pixel_link_loss = K.sum([pixel_pos_link_loss,
                                 pixel_neg_link_loss * pixel_link_neg_loss_weight_lambda])

    return pixel_cls_loss * pixel_cls_loss_weight_lambda + pixel_link_loss * pixel_link_loss_weight
