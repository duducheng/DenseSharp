import tensorflow as tf
from keras.layers import (Input, Conv3D, BatchNormalization, MaxPool3D, GlobalAveragePooling3D, Dropout, Dense,
                          Activation, Flatten, concatenate, add, Lambda)
from keras.models import Model
from keras.regularizers import l2


def spatial_red_block(in_layer, f, kernel_initializer, weight_decay):
    p1_layer1 = MaxPool3D(pool_size=(2, 2, 2),
                          padding='same')(in_layer)

    p2_layer1 = Conv3D(filters=f // 4, kernel_size=(3, 3, 3),
                       strides=(2, 2, 2), padding='same', use_bias=False,
                       kernel_initializer=kernel_initializer,
                       kernel_regularizer=l2(weight_decay))(in_layer)

    p3_layer1 = Conv3D(filters=f // 4, kernel_size=(1, 1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       kernel_regularizer=l2(weight_decay))(in_layer)
    p3_layer2 = Conv3D(filters=(f * 5) // 16, kernel_size=(3, 3, 3),
                       strides=(2, 2, 2), padding='same', use_bias=False,
                       kernel_initializer=kernel_initializer,
                       kernel_regularizer=l2(weight_decay))(p3_layer1)

    p4_layer1 = Conv3D(filters=f // 4, kernel_size=(1, 1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       kernel_regularizer=l2(weight_decay))(in_layer)
    p4_layer2 = Conv3D(filters=(f * 5) // 16, kernel_size=(3, 3, 3), padding='same',
                       kernel_initializer=kernel_initializer,
                       kernel_regularizer=l2(weight_decay))(p4_layer1)
    p4_layer3 = Conv3D(filters=(f * 7) // 16, kernel_size=(3, 3, 3),
                       strides=(2, 2, 2), padding='same', use_bias=False,
                       kernel_initializer=kernel_initializer,
                       kernel_regularizer=l2(weight_decay))(p4_layer2)

    unit = concatenate([p1_layer1, p2_layer1, p3_layer2, p4_layer3], axis=-1)

    bn = BatchNormalization(scale=False)(unit)

    return bn


def feat_red(in_layer, f, kernel_initializer, weight_decay):
    layer = Conv3D(filters=f // 2, kernel_size=(1, 1, 1),
                   padding='same', use_bias=False,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=l2(weight_decay))(in_layer)

    bn = BatchNormalization(scale=False)(layer)

    return bn


def res_conv_block(in_layer, f, n, kernel_initializer, weight_decay, activation):
    p1_layer1 = Conv3D(filters=n, kernel_size=(3, 3, 3), padding='same',
                       kernel_initializer=kernel_initializer,
                       kernel_regularizer=l2(weight_decay))(in_layer)

    p2_layer1 = Conv3D(filters=n, kernel_size=(1, 1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       kernel_regularizer=l2(weight_decay))(in_layer)
    p2_layer2 = Conv3D(filters=n, kernel_size=(3, 3, 3), padding='same',
                       kernel_initializer=kernel_initializer,
                       kernel_regularizer=l2(weight_decay))(p2_layer1)

    p3_layer1 = Conv3D(filters=n, kernel_size=(1, 1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       kernel_regularizer=l2(weight_decay))(in_layer)
    p3_layer2 = Conv3D(filters=n, kernel_size=(3, 3, 3), padding='same',
                       kernel_initializer=kernel_initializer,
                       kernel_regularizer=l2(weight_decay))(p3_layer1)
    p3_layer3 = Conv3D(filters=n, kernel_size=(3, 3, 3), padding='same',
                       kernel_initializer=kernel_initializer,
                       kernel_regularizer=l2(weight_decay))(p3_layer2)

    unit = concatenate([p1_layer1, p2_layer2, p3_layer3], axis=-1)
    unit = Conv3D(filters=f, kernel_size=(1, 1, 1),
                  padding='same', use_bias=False,
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer=l2(weight_decay))(unit)
    unit = add([in_layer, unit])

    bn = BatchNormalization(scale=False)(unit)

    relu = activation()(bn)

    return relu


def get_model(dhw=[32, 32, 32], weights=None, activation=lambda: Activation('relu'), final_kernel_size=5,
              kernel_initializer='he_uniform', weight_decay=0., scale=lambda x: x / 128. - 1., include_dense=True):
    shape = dhw + [1]
    inputs = Input(shape)
    if scale:
        scaled = Lambda(scale)(inputs)
    else:
        scaled = inputs

    layer0 = Conv3D(filters=64, kernel_size=(3, 3, 3),
                    padding='same', use_bias=False,
                    kernel_initializer=kernel_initializer)(scaled)
    layer0 = BatchNormalization(scale=False)(layer0)

    layer1 = spatial_red_block(layer0, 64, kernel_initializer, weight_decay)
    layer2 = res_conv_block(layer1, 128, 32, kernel_initializer,
                            weight_decay, activation)
    layer3 = spatial_red_block(layer2, 128, kernel_initializer, weight_decay)
    layer4 = res_conv_block(layer3, 256, 64, kernel_initializer,
                            weight_decay, activation)
    layer5 = spatial_red_block(layer4, 256, kernel_initializer, weight_decay)
    layer6 = res_conv_block(layer5, 512, 128, kernel_initializer,
                            weight_decay, activation)

    layer7 = feat_red(layer6, 512, kernel_initializer, weight_decay)
    layer8 = res_conv_block(layer7, 256, 64, kernel_initializer,
                            weight_decay, activation)
    layer9 = feat_red(layer8, 256, kernel_initializer, weight_decay)
    if include_dense:
        layer10 = Conv3D(filters=128, kernel_size=(final_kernel_size,) * 3, padding='valid',
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=l2(weight_decay))(layer9)
        layer10 = activation()(layer10)
        flatten = Flatten()(layer10)
        flatten = Dropout(rate=0.2)(flatten)
        fc = Dense(128, kernel_initializer=kernel_initializer,
                   kernel_regularizer=l2(weight_decay))(flatten)
        latest = activation()(fc)
    else:
        latest = activation()(layer9)
        latest = GlobalAveragePooling3D()(latest)

    outputs = Dense(4, activation='softmax', kernel_initializer=kernel_initializer,
                    kernel_regularizer=l2(weight_decay))(latest)

    model = Model(inputs=inputs, outputs=outputs)

    if weights is not None:
        model.load_weights(weights)

    return model


def get_compiled(dhw=[36, 36, 36], activation=lambda: Activation('relu'),
                 final_kernel_size=5, include_dense=True, scale=lambda x: x / 128. - 1.,
                 loss='categorical_crossentropy', gpu=0, optimizer='adam', weights=None,
                 kernel_initializer='he_uniform', weight_decay=0.):
    with tf.device('/gpu:%s' % gpu):
        model = get_model(dhw, weights, activation, final_kernel_size,
                          kernel_initializer, weight_decay, scale, include_dense)
        model.compile(optimizer=optimizer, loss=loss,
                      metrics=["categorical_accuracy", loss])
    model.summary()
    return model


if __name__ == '__main__':
    from keras.layers.advanced_activations import LeakyReLU

    model = get_compiled(activation=LeakyReLU)
