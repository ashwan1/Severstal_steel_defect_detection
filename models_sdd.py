import segmentation_models as sm
from classification_models.resnet import ResNet18, ResNet34
from efficientnet.keras import EfficientNetB0, EfficientNetB1, EfficientNetB3, EfficientNetB5
from keras import backend as K
from keras import layers, models, optimizers, utils
from keras.applications.mobilenet_v2 import MobileNetV2

from nn_blocks import aspp, conv, cbam, se_block


class SDDModel:
    def __init__(self, backbone, input_shape, lr, dropout_rate, model_arc, use_cbam=False, use_se=False,
                 n_classes=4, accum_steps=0, layer_lrs=None, use_multi_gpu=False, gpu_count=4):
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
        self.use_multi_gpu = use_multi_gpu
        self.gpu_count = gpu_count
        self.feature_layers = {
            # o/p shapes: [(64, 400, 64), (32, 200, 128), (16, 100, 256), (8, 50, 512)]
            'resnet18': ['stage2_unit1_relu1', 'stage3_unit1_relu1', 'stage4_unit1_relu1', 'relu1'],
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

    def _build_model(self):
        if self.model_arc == 'unet':
            self.model = sm.Unet(self.backbone_name, input_shape=self.input_shape, classes=self.n_classes,
                                 activation='sigmoid', encoder_weights='imagenet')
        elif self.model_arc == 'deeplab':
            self.model = self._get_deeplab_v3(use_cbam=self.use_cbam, use_se=self.use_se)
        elif self.model_arc == 'deeplab_classification_binary':
            self.model = self._get_deeplab_v3(use_cbam=self.use_cbam, classification=True)
        if self.use_multi_gpu:
            self.parallel_model = utils.multi_gpu_model(self.model, gpus=self.gpu_count, cpu_relocation=True)

    def _get_deeplab_v3(self, use_cbam=False, use_se=False, classification=False):
        img_height, img_width = self.input_shape[0], self.input_shape[1]
        c_o = None
        backbone_model = self._get_backbone_model()
        assert backbone_model is not None, f'backbone should be one of {list(self.feature_layers.keys())}'
        feature_layers = self.feature_layers[self.backbone_name]
        img_features = backbone_model.get_layer(feature_layers[2]).output
        if use_cbam:
            img_features = cbam(img_features)
        if classification:
            c = layers.concatenate([layers.GlobalAvgPool2D()(img_features),
                                    layers.GlobalMaxPool2D()(img_features)])
            c = layers.Dropout(self.dropout_rate)(c)
            c_o = layers.Dense(1, activation='sigmoid', name='classification_output')(c)
        x = aspp(img_features)
        h_t, w_t = K.int_shape(x)[1:3]
        scale = (img_height / 4) // h_t, (img_width / 4) // w_t
        x = layers.UpSampling2D(scale, interpolation='bilinear')(x)
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
        x = layers.SpatialDropout2D(self.dropout_rate)(x)
        x = conv(x, num_filters=128, kernel_size=3)
        if use_cbam:
            x = cbam(x)
        if use_se:
            x = se_block(x)
        h_t, w_t = K.int_shape(x)[1:3]
        scale = img_height // h_t, img_width // w_t
        x = layers.UpSampling2D(size=scale, interpolation='bilinear')(x)
        x = layers.Conv2D(self.n_classes, (1, 1))(x)
        o = layers.Activation('sigmoid', name='output_layer')(x)
        if classification:
            return models.Model(inputs=backbone_model.input, outputs=[c_o, o])
        else:
            return models.Model(inputs=backbone_model.input, outputs=o)

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
        if self.use_multi_gpu:
            self._compile_model(self.parallel_model, optimizer)
        else:
            self._compile_model(self.model, optimizer)

    def _compile_model(self, model, optimizer):
        if self.model_arc == 'deeplab_classification_binary':
            model.compile(optimizer, ['binary_crossentropy', sm.losses.bce_dice_loss],
                          metrics={
                              'classification_output': ['accuracy'],
                              'output_layer': [sm.metrics.iou_score, sm.metrics.f1_score]
                          })
        else:
            model.compile(optimizer, sm.losses.bce_dice_loss,
                          metrics=[sm.metrics.iou_score, sm.metrics.f1_score])

    def get_model(self):
        self._build_model()
        self._compile()
        if self.use_multi_gpu:
            return self.model, self.parallel_model
        else:
            return self.model
