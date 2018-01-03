# encoding=utf-8
from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras import backend as K


def conv_block(input_x, filters, blocks, kernel_size):
    channel_axis = -1 if K.image_data_format() == "channels_last" else 1

    if blocks<1:
        raise ValueError("blocks must be grater than 0")

    if isinstance(filters, list) and len(filters)!=blocks:
        raise ValueError("length of filters must equals blocks")

    f = filters[0] if isinstance(filters, list) else filters
    x = Conv2D(f, kernel_size)(input_x)
    x = BatchNormalization(axis=channel_axis)(x)

    if blocks > 1:
        for i in range(blocks-1):
            f = filters[i+1] if isinstance(filters, list) else filters
            x = Activation("relu")(x)
            x = Conv2D(f, kernel_size, padding="same")(x)
            x = BatchNormalization(axis=channel_axis)(x)

    f = filters[-1] if isinstance(filters, list) else filters
    concat_x = Conv2D(f, kernel_size)(input_x)
    concat_x = BatchNormalization(axis=channel_axis)(concat_x)

    x = layers.add([x, concat_x])
    x = Activation("relu")(x)
    return x


def identity_block(input_x, filters, blocks, kernel_size):
    channel_axis = -1 if K.image_data_format() == "channels_last" else 1

    if blocks < 1:
        raise ValueError("blocks must be grater than 0")

    if isinstance(filters, list) and len(filters) != blocks:
        raise ValueError("length of filters must equals blocks")

    f = filters[0] if isinstance(filters, list) else filters
    x = Conv2D(f, kernel_size, padding="same")(input_x)
    x = BatchNormalization(axis=channel_axis)(x)

    if blocks > 1:
        for i in range(blocks - 1):
            f = filters[i + 1] if isinstance(filters, list) else filters
            x = Activation("relu")(x)
            x = Conv2D(f, kernel_size, padding="same")(x)
            x = BatchNormalization(axis=channel_axis)(x)

    x = layers.add([x, input_x])
    x = Activation("relu")(x)
    return x


def ResNet(input_shape, classes=1):
    img_input = Input(shape=input_shape)

    channel_axis = 1 if K.image_data_format()=="channels_last" else -1

    x = Conv2D(64, 3)(img_input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2,2))(x)

    x = conv_block(x, [32, 64, 128], blocks=3, kernel_size=3)
    x = identity_block(x, [32, 64, 128], blocks=3, kernel_size=2)

    x = conv_block(x, [128, 256, 512], blocks=3, kernel_size=3)
    x = identity_block(x, [128, 256, 512], blocks=3, kernel_size=2)

    x = MaxPooling2D()(x)

    x = GlobalMaxPooling2D()(x)
    x = Dense(64, activation="relu")(x)

    if classes==1:
        x = Dense(1, activation="sigmoid")(x)
    else:
        x = Dense(classes, activation="softmax")(x)

    model = Model(img_input, x)
    return model