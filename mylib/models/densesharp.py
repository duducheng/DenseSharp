from keras.layers import (Conv3D, BatchNormalization, AveragePooling3D, concatenate, Lambda, SpatialDropout3D,
                          Activation, Input, GlobalAvgPool3D, Dense, Conv3DTranspose, add)
from keras.regularizers import l2 as l2_penalty
from keras.models import Model

from .metrics import precision, recall, fmeasure
from .losses import DiceLoss

PARAMS = {
    'activation': lambda: Activation('relu'),
    'bn_scale': True,
    'weight_decay': 0.,
    'kernel_initializer': 'he_uniform',
    'first_scale': lambda x: x / 128. - 1.,
    'dhw': [32, 32, 32],
    'k': 16,
    'bottleneck': 4,
    'compression': 2,
    'first_layer': 32,
    'down_structure': [4, 4, 4],
    'output_size': 1,
    'dropout_rate': None
}


def _conv_block(x, filters):
    bn_scale = PARAMS['bn_scale']
    activation = PARAMS['activation']
    kernel_initializer = PARAMS['kernel_initializer']
    weight_decay = PARAMS['weight_decay']
    bottleneck = PARAMS['bottleneck']
    dropout_rate = PARAMS['dropout_rate']

    x = BatchNormalization(scale=bn_scale, axis=-1)(x)
    x = activation()(x)
    x = Conv3D(filters * bottleneck, kernel_size=(1, 1, 1), padding='same', use_bias=False,
               kernel_initializer=kernel_initializer,
               kernel_regularizer=l2_penalty(weight_decay))(x)
    if dropout_rate is not None:
        x = SpatialDropout3D(dropout_rate)(x)
    x = BatchNormalization(scale=bn_scale, axis=-1)(x)
    x = activation()(x)
    x = Conv3D(filters, kernel_size=(3, 3, 3), padding='same', use_bias=True,
               kernel_initializer=kernel_initializer,
               kernel_regularizer=l2_penalty(weight_decay))(x)
    return x


def _dense_block(x, n):
    k = PARAMS['k']

    for _ in range(n):
        conv = _conv_block(x, k)
        x = concatenate([conv, x], axis=-1)
    return x


def _transmit_block(x, is_last):
    bn_scale = PARAMS['bn_scale']
    activation = PARAMS['activation']
    kernel_initializer = PARAMS['kernel_initializer']
    weight_decay = PARAMS['weight_decay']
    compression = PARAMS['compression']

    x = BatchNormalization(scale=bn_scale, axis=-1)(x)
    x = activation()(x)
    if is_last:
        x = GlobalAvgPool3D()(x)
    else:
        *_, f = x.get_shape().as_list()
        x = Conv3D(f // compression, kernel_size=(1, 1, 1), padding='same', use_bias=True,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=l2_penalty(weight_decay))(x)
        x = AveragePooling3D((2, 2, 2), padding='valid')(x)
    return x


def get_model(weights=None, verbose=True, **kwargs):
    for k, v in kwargs.items():
        assert k in PARAMS
        PARAMS[k] = v
    if verbose:
        print("Model hyper-parameters:", PARAMS)

    dhw = PARAMS['dhw']
    first_scale = PARAMS['first_scale']
    first_layer = PARAMS['first_layer']
    kernel_initializer = PARAMS['kernel_initializer']
    weight_decay = PARAMS['weight_decay']
    down_structure = PARAMS['down_structure']
    output_size = PARAMS['output_size']

    shape = dhw + [1]

    inputs = Input(shape=shape)

    if first_scale is not None:
        scaled = Lambda(first_scale)(inputs)
    else:
        scaled = inputs
    conv = Conv3D(first_layer, kernel_size=(3, 3, 3), padding='same', use_bias=True,
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer=l2_penalty(weight_decay))(scaled)

    downsample_times = len(down_structure)
    top_down = []
    for l, n in enumerate(down_structure):
        db = _dense_block(conv, n)
        top_down.append(db)
        conv = _transmit_block(db, l == downsample_times - 1)

    feat = top_down[-1]
    for top_feat in reversed(top_down[:-1]):
        *_, f = top_feat.get_shape().as_list()
        deconv = Conv3DTranspose(filters=f, kernel_size=2, strides=2, use_bias=True,
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=l2_penalty(weight_decay))(feat)
        feat = add([top_feat, deconv])
    seg_head = Conv3D(1, kernel_size=(1, 1, 1), padding='same',
                      activation='sigmoid', use_bias=True,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=l2_penalty(weight_decay),
                      name='seg')(feat)

    if output_size == 1:
        last_activation = 'sigmoid'
    else:
        last_activation = 'softmax'

    clf_head = Dense(output_size, activation=last_activation,
                     kernel_regularizer=l2_penalty(weight_decay),
                     kernel_initializer=kernel_initializer,
                     name='clf')(conv)

    model = Model(inputs, [clf_head, seg_head])
    if verbose:
        model.summary()

    if weights is not None:
        model.load_weights(weights)
    return model


def get_compiled(loss={"clf": 'binary_crossentropy',
                       "seg": DiceLoss()},
                 optimizer='adam',
                 metrics={'clf': ['accuracy', precision, recall, fmeasure],
                          'seg': [precision, recall, fmeasure]},
                 loss_weights={"clf": 1., "seg": .2}, weights=None, **kwargs):
    model = get_model(weights=weights, **kwargs)
    model.compile(loss=loss, optimizer=optimizer,
                  metrics=metrics, loss_weights=loss_weights)
    return model
