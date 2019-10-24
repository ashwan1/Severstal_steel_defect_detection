import os
import time
from pathlib import Path

import pandas as pd
import segmentation_models as sm
from keras import callbacks, utils, models, layers, optimizers
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from sklearn.model_selection import train_test_split

from custom_objects.callbacks import ObserveMetrics
from nn_blocks import conv
from utils.data_utils import DataSequence, prepare_data_df

_MODEL_ARC = 'ensemble'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

ex = Experiment(name=_MODEL_ARC)
ex.observers.append(MongoObserver.create(
    url='127.0.0.1:27017', db_name='severstal_sdd'))
ex.captured_out_filter = apply_backspaces_and_linefeeds


def train_model(model, train_seq, val_seq, training_callbacks):
    model.summary()
    history = model.fit_generator(train_seq, epochs=50, verbose=2, callbacks=training_callbacks,
                                  validation_data=val_seq, max_queue_size=4, workers=4,
                                  use_multiprocessing=True)
    return history


# noinspection PyUnusedLocal
@ex.config
def config():
    seed = 33
    seg1_path = 'artifacts/deeplab/resnet18/16-10-19_09:44/seg_model_best.h5'
    seg2_path = 'artifacts/deeplab/resnet34/17-10-19_13:19/seg_model_best.h5'
    batch_size = 8
    lr = 0.0005
    dropout_rate = 0.1
    data_path = 'data'
    artifacts_folder = f'artifacts/{_MODEL_ARC}/{time.strftime("%d-%m-%y_%H:%M", time.localtime())}'
    img_size = (256, 1600, 3)


@ex.automain
def run(seg1_path, seg2_path, batch_size, lr, dropout_rate, data_path, artifacts_folder,
        img_size, seed, _run):
    artifacts_folder = Path(artifacts_folder)
    artifacts_folder.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_path)
    data_df = pd.read_csv(data_path / 'train.csv')
    data_df = prepare_data_df(data_df)
    print(data_df.info())
    print(data_df.head(10))

    train_df, val_df = train_test_split(data_df, test_size=0.2, random_state=seed)
    train_df = train_df.sample(frac=1).reset_index(drop=True)

    ckpt_path = artifacts_folder / 'ckpts'
    ckpt_path.mkdir(exist_ok=True, parents=True)

    seg1_model = models.load_model(seg1_path, compile=False)
    for layer in seg1_model.layers:
        layer.name = f'seg1_{layer.name}'
        layer.trainable = False

    seg2_model = models.load_model(seg2_path, compile=False)
    for layer in seg2_model.layers:
        layer.name = f'seg2_{layer.name}'
        layer.trainable = False

    x = layers.concatenate([seg1_model.output, seg2_model.output])
    x = layers.SpatialDropout2D(dropout_rate)(x)
    x = conv(x, 16, 3)
    x = layers.Conv2D(4, (1, 1))(x)
    o = layers.Activation('sigmoid', name='output_layer')(x)
    segmentation_model = models.Model([seg1_model.input, seg2_model.input], o)
    segmentation_model.compile(optimizers.Adam(lr), sm.losses.bce_dice_loss,
                               metrics=[sm.metrics.iou_score, sm.metrics.f1_score])
    utils.plot_model(segmentation_model, str(artifacts_folder / 'seg_model.png'), show_shapes=True)

    training_callbacks = [
        callbacks.ReduceLROnPlateau(patience=3, verbose=1, min_lr=1e-7),
        callbacks.EarlyStopping(patience=5, verbose=1, restore_best_weights=True),
        callbacks.ModelCheckpoint(str(ckpt_path / 'seg_model-{epoch:04d}-{val_loss:.4f}.hdf5'),
                                  verbose=1, save_best_only=True),
        callbacks.TensorBoard(log_dir=str(artifacts_folder / 'tb_logs')),
        callbacks.TerminateOnNaN(),
        ObserveMetrics(_run, 'seg')
    ]

    train_seq = DataSequence(seed, train_df, batch_size, img_size, 'data/train_images', mode='train', shuffle=True,
                             augment=True, for_stacker=True)
    val_seq = DataSequence(seed, val_df, batch_size, img_size, 'data/train_images', mode='val', shuffle=False,
                           augment=False, for_stacker=True)

    history = train_model(segmentation_model, train_seq, val_seq, training_callbacks)
    models.save_model(segmentation_model, str(artifacts_folder / 'seg_model_best.h5'))
    segmentation_model.save_weights(str(artifacts_folder / 'weights_seg_model_best.h5'))

    print('loading model back')
    del segmentation_model
    segmentation_model = models.load_model(str(artifacts_folder / 'seg_model_best.h5'), compile=False)
    segmentation_model.predict_generator(val_seq, verbose=1)

    return history.history['val_loss'][-1]
