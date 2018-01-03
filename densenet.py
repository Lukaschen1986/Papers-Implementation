# encoding=utf-8
from keras.models import Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input, Concatenate
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras import backend as K


def conv_block(x, filters, kr, B=None, dropout=0.0, weight_decay=1e-4):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    if B is not None:
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=filters*B, kernel_size=(1,1), kernel_initializer='he_normal', padding='same', use_bias=False,
                   kernel_regularizer=l2(weight_decay))(x)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=filters, kernel_size=kr, kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)

    x = Dropout(dropout)(x)
    return x


def dense_block(x, layers, filters, kr, growth_rate, B=None, dropout=0.0, weight_decay=1e-4):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    xs = [x]

    for _ in range(layers):
        x = conv_block(x, growth_rate, kr, B, dropout, weight_decay)
        xs.append(x)
        x = Concatenate(axis=channel_axis)(xs)
        filters += growth_rate

    return x, filters


def transition(x, filters, C=1.0, dropout=0.0, weight_decay=1e-4):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=int(filters*C), kernel_size=(1, 1), kernel_initializer='he_normal', padding='same',
                   use_bias=False, kernel_regularizer=l2(weight_decay))(x)

    x = Dropout(dropout)(x)

    x = AveragePooling2D()(x)
    return x


def DenseNet(input_shape, classes, filters, growth_rate, init_kernel_size=(3,3), blocks=3, block_layers=2, block_kernel_size=3, B=4, C=1.0, dropout=0.0, weight_decay=1e-4):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    if blocks<1:
        raise ValueError("blocks must be grater than 0")

    if isinstance(block_layers, list) and len(block_layers)!=blocks:
        raise ValueError("length of filters must equals blocks")

    if isinstance(block_kernel_size, list) and len(block_kernel_size)!=blocks:
        raise ValueError("length of kernel_size must equals blocks")

    img_input = Input(input_shape)

    x = Conv2D(filters, init_kernel_size, kernel_initializer="he_uniform", padding="same", use_bias=False,
               kernel_regularizer=l2(weight_decay))(img_input)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    # x = AveragePooling2D((2, 2))(x)

    for i in range(blocks-1):
        la=block_layers[i] if isinstance(block_layers, list) else block_layers
        ks=block_kernel_size[i] if isinstance(block_kernel_size, list) else block_kernel_size
        x, filters = dense_block(x, la, filters, ks, growth_rate, B, dropout, weight_decay)
        x = transition(x, filters, C, dropout, weight_decay)
        filters = int(filters*C)

    la = block_layers[-1] if isinstance(block_layers, list) else block_layers
    ks = block_kernel_size[-1] if isinstance(block_kernel_size, list) else block_kernel_size
    x, _ = dense_block(x, la, filters, ks, growth_rate, B, dropout, weight_decay)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = AveragePooling2D((2, 2))(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation="relu")(x)

    if classes==1:
        x = Dense(1, activation="sigmoid")(x)
    else:
        x = Dense(classes, activation="softmax")(x)

    model = Model(img_input, x)
    return model