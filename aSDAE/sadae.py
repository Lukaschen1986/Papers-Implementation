# encoding=utf-8

import keras
from keras.layers import Dense, BatchNormalization, Activation, Add, Input, Dot
from keras.layers.noise import GaussianNoise
from keras.models import Model
import tensorflow as tf


def aSDAE(S, X, latent_dim, hidden_unit_nums=[], stddev=0.1, lam=0.5):
    S_noise = GaussianNoise(stddev)(S)
    X_noise = GaussianNoise(stddev)(X)
    h = S_noise

    for num in hidden_unit_nums:
        h_s = Dense(num, kernel_regularizer=keras.regularizers.l2(lam),
                    bias_regularizer=keras.regularizers.l2(lam))(h)
        h_x = Dense(num, kernel_regularizer=keras.regularizers.l2(lam),
                    bias_regularizer=keras.regularizers.l2(lam))(X_noise)
        h = Add()([h_s, h_x])
        h = Activation("relu")(h)

    latent_s = Dense(latent_dim, kernel_regularizer=keras.regularizers.l2(lam),
                     bias_regularizer=keras.regularizers.l2(lam))(h)
    latent_x = Dense(latent_dim, kernel_regularizer=keras.regularizers.l2(lam),
                     bias_regularizer=keras.regularizers.l2(lam))(X_noise)
    latent = Add()([latent_s, latent_x])
    latent = Activation("relu")(latent)
    h = latent

    for num in hidden_unit_nums[::-1]:
        h_s = Dense(num, kernel_regularizer=keras.regularizers.l2(lam),
                    bias_regularizer=keras.regularizers.l2(lam))(h)
        h_x = Dense(num, kernel_regularizer=keras.regularizers.l2(lam),
                    bias_regularizer=keras.regularizers.l2(lam))(X_noise)
        h = Add()([h_s, h_x])
        h = Activation("relu")(h)

    S_ = Dense(int(S.shape[1]), kernel_regularizer=keras.regularizers.l2(lam),
               bias_regularizer=keras.regularizers.l2(lam))(h)
    X_ = Dense(int(X.shape[1]), kernel_regularizer=keras.regularizers.l2(lam),
               bias_regularizer=keras.regularizers.l2(lam))(h)
    S_ = Activation("relu")(S_)
    X_ = Activation("relu")(X_)

    return latent, S_, X_


def aSDAE_reSys(user_shape, user_add_shape, item_shape, item_add_shape, latent_dim, user_hun=[], item_hun=[],
                alpha_1=0.5, alpha_2=0.5, lam=0.5):
    S_U = Input(shape=user_shape)
    X_U = Input(shape=user_add_shape)
    S_I = Input(shape=item_shape)
    X_I = Input(shape=item_add_shape)
    U_latent, S_U_, X_U_ = aSDAE(S_U, X_U, latent_dim, user_hun, lam=lam)
    I_latent, S_I_, X_I_ = aSDAE(S_I, X_I, latent_dim, item_hun, lam=lam)

    y_ = Dot(axes=1)([U_latent, I_latent])

    s_u = tf.reduce_sum(tf.square(tf.subtract(S_U,S_U_)))
    x_u = tf.reduce_sum(tf.square(tf.subtract(X_U,X_U_)))
    s_i = tf.reduce_sum(tf.square(tf.subtract(S_I,S_I_)))
    x_i = tf.reduce_sum(tf.square(tf.subtract(X_I,X_I_)))
    u = tf.reduce_sum(tf.square(U_latent))
    i = tf.reduce_sum(tf.square(I_latent))
    loss = lambda y_true,y_pre: tf.reduce_sum(tf.square(y_true-y_pre)) \
                                + alpha_1*s_u + (1-alpha_1)*x_u \
                                + alpha_2*s_i + (1-alpha_2)*x_i \
                                + lam*(u+i)

    model = Model(inputs=[S_U,X_U,S_I,X_I], outputs=y_)

    return model, loss
