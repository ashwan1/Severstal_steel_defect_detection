from keras import layers
from keras import backend as K


def conv(tensor, num_filters, kernel_size, padding='same', strides=1, dilation_rate=1):
    x = layers.Conv2D(filters=num_filters, kernel_size=kernel_size, padding=padding, strides=strides,
                      dilation_rate=dilation_rate)(tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def separable_conv(tensor, num_filters, kernel_size, padding='same', strides=1, dilation_rate=1):
    x = layers.SeparableConv2D(filters=num_filters, depth_multiplier=1, kernel_size=kernel_size, padding=padding,
                               strides=strides, dilation_rate=dilation_rate, )(tensor)
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
        ym.append(separable_conv(yc, 512, 3, dilation_rate=rate))
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
