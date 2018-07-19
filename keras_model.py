import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, SeparableConv2D, BatchNormalization, AveragePooling2D
from keras.layers import Softmax, UpSampling2D, Input, concatenate, merge
# from keras.layers.merge import add


class PixelLink:
    def __init__(self):
        self.net = Sequential()
        self.width_multiplier = 1
        # will change
        self.input_shape = (640, 360, 3)
        self.txt_net, self.plx_net = self.create_model()

    def add(self, model1, model2):
        shape1 = model1.input_shape
        print(shape1)

        shape2 = model2.input_shape
        print(shape2)
        iL = [keras.layers.Input(shape=shape1), keras.layers.Input(shape=shape2)]
        hL = [model1(iL[0]), model2(iL[1])]
        oL = keras.layers.Add()(hL)
        model3 = keras.models.Model(inputs=iL, outputs=oL)
        return model3


    # create pixel link model
    def create_model(self, num_class=1000):

        self.net.add(Conv2D(filters=round(self.width_multiplier * 32), kernel_size=[3, 3],
                            input_shape=self.input_shape))
        self._add_df_layer(self.net, 64)
        self._add_df_layer(self.net, 128, downsample=True)
        self._add_df_layer(self.net, 128)

        # two different net, so there are two different convolution layer
        shortcut1 = self._shortcut(self.net)
        shortcut_link1 = self._shortcut(self.net, 16)

        self._add_df_layer(self.net, 256, downsample=True)
        self._add_df_layer(self.net, 256)

        shortcut2 = self._shortcut(self.net)
        shortcut_link2 = self._shortcut(self.net, 16)

        self._add_df_layer(self.net, 512, downsample=True)

        self._add_df_layer(self.net, 512)
        self._add_df_layer(self.net, 512)
        self._add_df_layer(self.net, 512)
        self._add_df_layer(self.net, 512)
        self._add_df_layer(self.net, 512)

        shortcut3 = self._shortcut(self.net)
        shortcut_link3 = self._shortcut(self.net, 16)

        self._add_df_layer(self.net, 1024, downsample=True)
        self._add_df_layer(self.net, 1024)

        # the net decide text or not
        text_net = self._shortcut(self.net)
        text_net = self._up_sample(text_net, shortcut3)
        text_net = self._up_sample(text_net, shortcut2)
        text_net = self._up_sample(text_net, shortcut1)
        text_net.add(Softmax(text_net))

        # the net decide pixel linked or not
        pixel_net = self._shortcut(self.net)
        pixel_net = self._up_sample(pixel_net, shortcut_link3)
        pixel_net = self._up_sample(pixel_net, shortcut_link2)
        pixel_net = self._up_sample(pixel_net, shortcut_link1)
        pixel_net.add(Softmax(pixel_net))

        # last three layers of mobilenet, we may neglect this stuff
        # self.net.add(AveragePooling2D(pool_size=[7, 7]))
        # self.net.add(Dense(num_class))
        # self.net.add(Softmax())

        return text_net, pixel_net

    def _up_sample(self, net, shortcut):
        net.add(UpSampling2D(size=(2, 2)))
        net = self.add(net, shortcut)
        return net

    def _shortcut(self, _input, num_pwc_filter=2):
        shortcut = _input
        shortcut.add(Conv2D(filters=num_pwc_filter, kernel_size=[1, 1]))
        return shortcut

    def _add_df_layer(self, net, num_pwc_filters, downsample=False):
        num_pwc_filters = round(num_pwc_filters * self.width_multiplier)
        _stride = 2 if downsample else 1

        net.add(SeparableConv2D(filters=num_pwc_filters, kernel_size=[3, 3],
                strides=_stride))
        net.add(BatchNormalization())
        net.add(Conv2D(kernel_size=[1, 1], filters=num_pwc_filters))
        net.add(BatchNormalization())




