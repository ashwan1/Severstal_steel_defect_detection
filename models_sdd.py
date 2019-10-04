import segmentation_models as sm
from classification_models.resnet import ResNet18
from keras import backend as K
from keras import layers, models, optimizers, utils
from keras.applications.mobilenet_v2 import MobileNetV2

from nn_blocks import aspp, conv, cbam


class SDDModel:
    def __init__(self, backbone, input_shape, lr, dropout_rate, model_arc, n_classes=4,
                 accum_steps=0, layer_lrs=None, use_multi_gpu=False, gpu_count=4):
        self.backbone_name = backbone
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.lr = lr
        self.dropout_rate = dropout_rate
        self.model_arc = model_arc
        self.accum_steps = accum_steps
        self.layer_lrs = layer_lrs
        self.model = None
        self.use_multi_gpu = use_multi_gpu
        self.gpu_count = gpu_count
        self.feature_layers = {
            # o/p shapes: [(64, 400, 64), (32, 200, 128), (16, 100, 256), (8, 50, 512)]
            'resnet18': ['stage2_unit1_relu1', 'stage3_unit1_relu1', 'stage4_unit1_relu1', 'relu1'],
            # o/p shapes: [(64, 400, 144), (32, 200, 192), (16, 100, 576), (8, 50, 1280)]
            'mobilenetv2': ['block_3_expand_relu', 'block_6_expand_relu', 'block_13_expand_relu', 'out_relu']
        }

    def _build_model(self):
        if self.model_arc == 'unet':
            self.model = sm.Unet(self.backbone_name, input_shape=self.input_shape, classes=self.n_classes,
                                 activation='sigmoid', encoder_weights='imagenet')
        elif self.model_arc == 'deeplab':
            self.model = self._get_deeplab_v3()
        elif self.model_arc == 'deeplab_cbam':
            self.model = self._get_deeplab_v3(use_cbam=True)

        if self.use_multi_gpu:
            self.parallel_model = utils.multi_gpu_model(self.model, gpus=self.gpu_count, cpu_relocation=True)

    def _get_deeplab_v3(self, use_cbam=False):
        img_height, img_width = self.input_shape[0], self.input_shape[1]
        backbone_model = None
        if self.backbone_name == 'resnet18':
            backbone_model = ResNet18(input_shape=self.input_shape, include_top=False, weights='imagenet')
        elif self.backbone_name == 'mobilenetv2':
            backbone_model = MobileNetV2(input_shape=self.input_shape, include_top=False, weights='imagenet')
        assert backbone_model is not None, f'backbone should be one of {list(self.feature_layers.keys())}'
        feature_layers = self.feature_layers[self.backbone_name]
        img_features = backbone_model.get_layer(feature_layers[2]).output
        if use_cbam:
            img_features = cbam(img_features)
        x = aspp(img_features)
        h_t, w_t = K.int_shape(x)[1:3]
        scale = (img_height / 4) // h_t, (img_width / 4) // w_t
        x = layers.UpSampling2D(scale, interpolation='bilinear')(x)
        y = conv(backbone_model.get_layer(feature_layers[0]).output, 64, 1)
        x = layers.concatenate([x, y])
        x = conv(x, num_filters=128, kernel_size=3)
        x = layers.SpatialDropout2D(self.dropout_rate)(x)
        x = conv(x, num_filters=128, kernel_size=3)
        h_t, w_t = K.int_shape(x)[1:3]
        scale = img_height // h_t, img_width // w_t
        x = layers.UpSampling2D(size=scale, interpolation='bilinear')(x)
        x = layers.Conv2D(self.n_classes, (1, 1))(x)
        o = layers.Activation('sigmoid', name='output_layer')(x)
        return models.Model(inputs=backbone_model.input, outputs=o)

    def _compile(self):
        optimizer = optimizers.Adam(lr=self.lr)
        if self.use_multi_gpu:
            self.parallel_model.compile(optimizer, sm.losses.bce_dice_loss,
                                        metrics=[sm.metrics.iou_score, sm.metrics.f1_score])
        else:
            self.model.compile(optimizer, sm.losses.bce_dice_loss,
                               metrics=[sm.metrics.iou_score, sm.metrics.f1_score])

    def get_model(self):
        self._build_model()
        self._compile()
        if self.use_multi_gpu:
            return self.model, self.parallel_model
        else:
            return self.model
