from keras import layers
from keras import backend as K


def _separable_conv(tensor, num_filters, kernel_size, padding='same', strides=1, dilation_rate=1):
    x = layers.SeparableConv2D(filters=num_filters, depth_multiplier=1, kernel_size=kernel_size, padding=padding,
                               strides=strides, dilation_rate=dilation_rate, )(tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def _channel_attention(input_feature, ratio=8):
    channel_dim = K.int_shape(input_feature)[-1]

    shared_layer_one = layers.Dense(channel_dim // ratio, activation='relu')
    shared_layer_two = layers.Dense(channel_dim)

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel_dim))(avg_pool)
    assert K.int_shape(avg_pool)[1:] == (1, 1, channel_dim)
    avg_pool = shared_layer_one(avg_pool)
    assert K.int_shape(avg_pool)[1:] == (1, 1, channel_dim // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert K.int_shape(avg_pool)[1:] == (1, 1, channel_dim)

    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel_dim))(max_pool)
    assert K.int_shape(max_pool)[1:] == (1, 1, channel_dim)
    max_pool = shared_layer_one(max_pool)
    assert K.int_shape(max_pool)[1:] == (1, 1, channel_dim // ratio)
    max_pool = shared_layer_two(max_pool)
    assert K.int_shape(max_pool)[1:] == (1, 1, channel_dim)

    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Activation('sigmoid')(cbam_feature)

    return layers.multiply([input_feature, cbam_feature])


def _spatial_attention(input_feature):
    kernel_size = 7
    cbam_feature = input_feature

    avg_pool = layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert K.int_shape(avg_pool)[-1] == 1
    max_pool = layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert K.int_shape(max_pool)[-1] == 1
    concat = layers.Concatenate(axis=3)([avg_pool, max_pool])
    assert K.int_shape(concat)[-1] == 2
    cbam_feature = layers.Conv2D(filters=1,
                                 kernel_size=kernel_size,
                                 strides=1,
                                 padding='same',
                                 activation='sigmoid')(concat)
    assert K.int_shape(cbam_feature)[-1] == 1

    return layers.multiply([input_feature, cbam_feature])


def conv(tensor, num_filters, kernel_size, padding='same', strides=1, dilation_rate=1):
    x = layers.Conv2D(filters=num_filters, kernel_size=kernel_size, padding=padding, strides=strides,
                      dilation_rate=dilation_rate)(tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def jpu(in_layers_output, out_channels=512):
    n_in_layers = len(in_layers_output)
    h, w = K.int_shape(in_layers_output[0])[1:3]
    for i in range(0, n_in_layers):
        in_layers_output[i] = conv(in_layers_output[i], out_channels, 3)
        if i != 0:
            h_t, w_t = K.int_shape(in_layers_output[i])[1:3]
            scale = (h // h_t, w // w_t)
            in_layers_output[i] = layers.UpSampling2D(scale, interpolation='bilinear')(in_layers_output[i])
    yc = layers.Concatenate()(in_layers_output)
    ym = []
    for rate in [1, 2, 4, 8]:
        ym.append(_separable_conv(yc, 512, 3, dilation_rate=rate))
    y = layers.Concatenate()(ym)
    return y


def aspp(tensor):
    dims = K.int_shape(tensor)
    y_pool = layers.AveragePooling2D(pool_size=(dims[1], dims[2]), name='average_pooling')(tensor)
    h_t, w_t = K.int_shape(y_pool)[1:3]
    scale = dims[1] // h_t, dims[2] // w_t
    y_pool = layers.UpSampling2D(size=scale, interpolation='bilinear')(y_pool)
    y_1 = conv(tensor, num_filters=128, kernel_size=1, dilation_rate=1)
    y_6 = conv(tensor, num_filters=128, kernel_size=3, dilation_rate=6)
    y_6.set_shape([None, dims[1], dims[2], 128])
    y_12 = conv(tensor, num_filters=128, kernel_size=3, dilation_rate=12)
    y_12.set_shape([None, dims[1], dims[2], 128])
    y_18 = conv(tensor, num_filters=128, kernel_size=3, dilation_rate=18)
    y_18.set_shape([None, dims[1], dims[2], 128])
    y = layers.Concatenate(axis=-1)([y_pool, y_1, y_6, y_12, y_18])
    y = conv(y, num_filters=128, kernel_size=1)
    return y


def cbam(tensor, ratio=8):
    cbam_feature = _channel_attention(tensor, ratio)
    cbam_feature = _spatial_attention(cbam_feature)
    return cbam_feature
