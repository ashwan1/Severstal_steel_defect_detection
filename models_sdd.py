import segmentation_models as sm
from classification_models.resnet import ResNet18, ResNet34
from efficientnet.keras import EfficientNetB0, EfficientNetB1, EfficientNetB3, EfficientNetB5
from keras import backend as K
from keras import layers, models, optimizers, metrics
from keras.applications.mobilenet_v2 import MobileNetV2

from nn_blocks import aspp, conv, cbam, se_block
from utils.model_utils import insert_layer_nonseq, mish_layer_factory

_FEATURE_LAYERS = {
            # o/p shapes: [(64, 400, 64), (32, 200, 128), (16, 100, 256), (8, 50, 512)]
            'resnet18': ['stage2_unit1_relu1', 'stage3_unit1_relu1', 'stage4_unit1_relu1', 'relu1_lambda_18'],
            # o/p shapes: [(64, 400, 64), (32, 200, 128), (16, 100, 256), (8, 50, 512)]
            'resnet34': ['stage2_unit1_relu1', 'stage3_unit1_relu1', 'stage4_unit1_relu1', 'relu1'],
            # o/p shapes: [(64, 400, 144), (32, 200, 192), (16, 100, 576), (8, 50, 1280)]
            'mobilenetv2': ['block_3_expand_relu', 'block_6_expand_relu', 'block_13_expand_relu', 'out_relu'],
            # o/p shapes: [(64, 400, 144), (32, 200, 240), (16, 100, 672), (8, 50, 1280)]
            'efficientnetb0': ['block3a_expand_activation', 'block4a_expand_activation',
                               'block6a_expand_activation', 'top_activation'],
            # o/p shapes: [(64, 400, 144), (32, 200, 240), (16, 100, 672), (8, 50, 1280)]
            'efficientnetb1': ['block3a_expand_activation', 'block4a_expand_activation',
                               'block6a_expand_activation', 'top_activation'],
            # o/p shapes: [(64, 400, 192), (32, 200, 288), (16, 100, 816), (8, 50, 1536)]
            'efficientnetb3': ['block3a_expand_activation', 'block4a_expand_activation',
                               'block6a_expand_activation', 'top_activation'],
            # o/p shapes: [(64, 400, 240), (32, 200, 384), (16, 100, 1056), (8, 50, 2048)]
            'efficientnetb5': ['block3a_expand_activation', 'block4a_expand_activation',
                               'block6a_expand_activation', 'top_activation']
        }


class SegmentationModel:
    def __init__(self, backbone, input_shape, lr, dropout_rate, model_arc, use_cbam=False,
                 use_se=False, n_classes=4, accum_steps=0, layer_lrs=None, cfn_model=None,
                 cfn_backbone=None, use_transpose_conv=False):
        self.backbone_name = backbone
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.lr = lr
        self.dropout_rate = dropout_rate
        self.model_arc = model_arc
        self.use_cbam = use_cbam
        self.use_se = use_se
        self.accum_steps = accum_steps
        self.layer_lrs = layer_lrs
        self.model = None
        self.cfn_model = cfn_model
        self.cfn_backbone = cfn_backbone
        self.use_transpose_conv = use_transpose_conv

    def _build_model(self):
        if self.model_arc == 'unet':
            self.model = sm.Unet(self.backbone_name, input_shape=self.input_shape, classes=self.n_classes,
                                 activation='sigmoid', encoder_weights='imagenet')
        elif self.model_arc == 'deeplab':
            self.model = self._get_deeplab_v3(use_cbam=self.use_cbam, use_se=self.use_se)

    def _get_deeplab_v3(self, use_cbam=False, use_se=False):
        img_height, img_width = self.input_shape[0], self.input_shape[1]
        backbone_model = self._get_backbone_model()
        assert backbone_model is not None, f'backbone should be one of {list(_FEATURE_LAYERS.keys())[:3]}'
        feature_layers = _FEATURE_LAYERS[self.backbone_name]
        img_features = backbone_model.get_layer(feature_layers[2]).output
        if use_cbam:
            img_features = cbam(img_features)
        if self.cfn_model is not None:
            img_features = self._concatenate_cfn_features(img_features, backbone_model)
        x = aspp(img_features)
        h_t, w_t, c_t = K.int_shape(x)[1:]
        scale = int((img_height / 4) // h_t), int((img_width / 4) // w_t)
        x = self._upsample_features(x, scale, c_t)
        y = conv(backbone_model.get_layer(feature_layers[0]).output, 64, 1)
        if use_cbam:
            y = cbam(y)
        if use_se:
            y = se_block(y)
        x = layers.concatenate([x, y])
        x = conv(x, num_filters=128, kernel_size=3)
        x = layers.SpatialDropout2D(self.dropout_rate)(x)
        x = conv(x, num_filters=128, kernel_size=3)
        if use_cbam:
            x = cbam(x)
        if use_se:
            x = se_block(x)
        h_t, w_t, c_t = K.int_shape(x)[1:]
        scale = img_height // h_t, img_width // w_t
        x = self._upsample_features(x, scale, c_t)
        x = layers.Conv2D(self.n_classes, (1, 1))(x)
        o = layers.Activation('sigmoid', name='output_layer')(x)
        return models.Model(inputs=backbone_model.input, outputs=o)

    def _concatenate_cfn_features(self, tensor, backbone_model):
        cfn_model = models.Model(inputs=self.cfn_model.input,
                                 outputs=self.cfn_model.get_layer(_FEATURE_LAYERS[self.cfn_backbone][-1]).output)
        for layer in cfn_model.layers:
            layer.trainable = False
        dims = K.int_shape(tensor)
        cfn_features = cfn_model(backbone_model.input)
        h_t, w_t, c_t = K.int_shape(cfn_features)[1:]
        scale = dims[1] // h_t, dims[2] // w_t
        x = self._upsample_features(cfn_features, scale, c_t)
        x = conv(x, num_filters=512, kernel_size=1)
        x = layers.concatenate([tensor, x])
        x = layers.Dropout(self.dropout_rate)(x)
        return x

    def _upsample_features(self, features, scale, n_channels):
        if self.use_transpose_conv:
            features = layers.Conv2DTranspose(n_channels, 3, strides=scale, padding='same')(features)
        else:
            features = layers.UpSampling2D(scale, interpolation='bilinear')(features)
        return features

    def _get_backbone_model(self):
        backbone_model = None
        if self.backbone_name == 'resnet18':
            backbone_model = ResNet18(input_shape=self.input_shape, include_top=False, weights='imagenet')
        if self.backbone_name == 'resnet34':
            backbone_model = ResNet34(input_shape=self.input_shape, include_top=False, weights='imagenet')
        elif self.backbone_name == 'mobilenetv2':
            backbone_model = MobileNetV2(input_shape=self.input_shape, include_top=False, weights='imagenet')
        return backbone_model

    def _compile(self):
        optimizer = optimizers.Adam(lr=self.lr)
        self.model.compile(optimizer, sm.losses.bce_dice_loss,
                           metrics=[sm.metrics.iou_score, sm.metrics.f1_score])

    def get_model(self):
        self._build_model()
        self.model = insert_layer_nonseq(self.model, '.*relu.*', mish_layer_factory, position='replace')
        self._compile()
        return self.model


class ClassificationModel:
    def __init__(self, backbone, input_shape, lr, n_classes=4):
        self.backbone_name = backbone
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.lr = lr
        self.model = None

    def _build_model(self):
        backbone_model = self._get_backbone_model()
        assert backbone_model is not None, f'backbone should be one of {list(_FEATURE_LAYERS.keys())}'
        x = layers.concatenate([layers.GlobalAvgPool2D()(backbone_model.output),
                                layers.GlobalMaxPool2D()(backbone_model.output)])
        o = layers.Dense(self.n_classes, activation='sigmoid', name='classification_output')(x)
        self.model = models.Model(inputs=backbone_model.input, outputs=o)

    def _get_backbone_model(self):
        backbone_model = None
        if self.backbone_name == 'resnet18':
            backbone_model = ResNet18(input_shape=self.input_shape, include_top=False, weights='imagenet')
        if self.backbone_name == 'resnet34':
            backbone_model = ResNet34(input_shape=self.input_shape, include_top=False, weights='imagenet')
        elif self.backbone_name == 'mobilenetv2':
            backbone_model = MobileNetV2(input_shape=self.input_shape, include_top=False, weights='imagenet')
        elif self.backbone_name == 'efficientnetb0':
            backbone_model = EfficientNetB0(input_shape=self.input_shape, include_top=False, weights='imagenet')
        elif self.backbone_name == 'efficientnetb1':
            backbone_model = EfficientNetB1(input_shape=self.input_shape, include_top=False, weights='imagenet')
        elif self.backbone_name == 'efficientnetb3':
            backbone_model = EfficientNetB3(input_shape=self.input_shape, include_top=False, weights='imagenet')
        elif self.backbone_name == 'efficientnetb5':
            backbone_model = EfficientNetB5(input_shape=self.input_shape, include_top=False, weights='imagenet')
        return backbone_model

    def _compile(self):
        optimizer = optimizers.Adam(lr=self.lr)
        self.model.compile(optimizer, 'binary_crossentropy',
                           metrics=[metrics.binary_accuracy, metrics.mse])

    def get_model(self):
        self._build_model()
        self.model = insert_layer_nonseq(self.model, '.*relu.*', mish_layer_factory, position='replace')
        self._compile()
        return self.model
