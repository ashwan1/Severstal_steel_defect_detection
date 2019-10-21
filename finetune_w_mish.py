import os
import time
from pathlib import Path

import pandas as pd
import segmentation_models as sm
from keras import callbacks, models, optimizers, metrics
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from sklearn.model_selection import train_test_split

from custom_objects.callbacks import ObserveMetrics
from utils.data_utils import DataSequence, prepare_data_df, ClassificationDataSeq
from utils.model_utils import insert_layer_nonseq, mish_layer_factory

_MODEL_ARC = 'deeplab_mish'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

ex = Experiment(name=_MODEL_ARC)
ex.observers.append(MongoObserver.create(
    url='127.0.0.1:27017', db_name='severstal_sdd'))
ex.captured_out_filter = apply_backspaces_and_linefeeds


def train_model(model, train_seq, val_seq, training_callbacks):
    model.summary()
    history = model.fit_generator(train_seq, epochs=100, verbose=2, callbacks=training_callbacks,
                                  validation_data=val_seq, max_queue_size=4, workers=4,
                                  use_multiprocessing=True)
    return history


# noinspection PyUnusedLocal
@ex.config
def config():
    seed = 33
    cfn_model_path = 'artifacts/deeplab/resnet18/16-10-19_09:44/cfn_model_best.h5'
    seg_model_path = 'artifacts/deeplab/resnet18/16-10-19_09:44/seg_model_best.h5'
    cfn_batch_multiplier = 3
    batch_size = 4
    lr = 5e-6
    data_path = 'data'
    artifacts_folder = f'artifacts/{_MODEL_ARC}/{time.strftime("%d-%m-%y_%H:%M", time.localtime())}'
    img_size = (256, 1600, 3)


@ex.automain
def run(cfn_model_path, seg_model_path, batch_size, lr, data_path, artifacts_folder,
        img_size, cfn_batch_multiplier, seed, _run):
    artifacts_folder = Path(artifacts_folder)
    artifacts_folder.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_path)
    data_df = pd.read_csv(data_path / 'train.csv')
    data_df = prepare_data_df(data_df)
    print(data_df.info())
    print(data_df.head(10))

    train_df, val_df = train_test_split(data_df, test_size=0.2, random_state=seed)
    print(f'length of train and val data before mix-match: {len(train_df.index)}, {len(val_df.index)}')

    ckpt_path = artifacts_folder / 'ckpts'
    ckpt_path.mkdir(exist_ok=True, parents=True)

    classification_model = models.load_model(cfn_model_path, compile=False)
    classification_model = insert_layer_nonseq(classification_model, '.*relu.*|.*re_lu.*', mish_layer_factory,
                                               position='replace')
    optimizer = optimizers.Adam(lr=lr)
    classification_model.compile(optimizer, 'binary_crossentropy',
                                 metrics=[metrics.binary_accuracy, metrics.mse])
    training_callbacks = [
        callbacks.ReduceLROnPlateau(patience=3, verbose=1, min_lr=1e-7, factor=0.5),
        callbacks.EarlyStopping(patience=5, verbose=1, restore_best_weights=True),
        callbacks.ModelCheckpoint(str(ckpt_path / 'cfn_model-{epoch:04d}-{val_loss:.4f}.hdf5'),
                                  verbose=1, save_best_only=True),
        callbacks.TensorBoard(log_dir=str(artifacts_folder / 'tb_logs')),
        callbacks.TerminateOnNaN(),
        ObserveMetrics(_run, 'cfn')
    ]
    train_seq = ClassificationDataSeq(seed, train_df, batch_size * cfn_batch_multiplier, img_size,
                                      'data/train_images', mode='train',
                                      shuffle=True, augment=True)
    val_seq = ClassificationDataSeq(seed, val_df, batch_size * cfn_batch_multiplier, img_size,
                                    'data/train_images', mode='val',
                                    shuffle=False, augment=False)
    train_model(classification_model, train_seq, val_seq, training_callbacks)
    models.save_model(classification_model, str(artifacts_folder / 'cfn_model_best.h5'))

    segmentation_model = models.load_model(seg_model_path, compile=False)
    segmentation_model = insert_layer_nonseq(segmentation_model, '.*relu.*|.*re_lu.*', mish_layer_factory,
                                             position='replace')
    optimizer = optimizers.Adam(lr=lr)
    segmentation_model.compile(optimizer, sm.losses.bce_dice_loss,
                               metrics=[sm.metrics.iou_score, sm.metrics.f1_score])

    training_callbacks = [
        callbacks.ReduceLROnPlateau(patience=3, verbose=1, min_lr=1e-7, factor=0.5),
        callbacks.EarlyStopping(patience=5, verbose=1, restore_best_weights=True),
        callbacks.ModelCheckpoint(str(ckpt_path / 'seg_model-{epoch:04d}-{val_loss:.4f}.hdf5'),
                                  verbose=1, save_best_only=True),
        callbacks.TensorBoard(log_dir=str(artifacts_folder / 'tb_logs')),
        callbacks.TerminateOnNaN(),
        ObserveMetrics(_run, 'seg')
    ]

    train_seq = DataSequence(seed * 2, train_df, batch_size, img_size, 'data/train_images', mode='train', shuffle=True,
                             augment=True)
    val_seq = DataSequence(seed * 2, val_df, batch_size, img_size, 'data/train_images', mode='val', shuffle=False,
                           augment=False)

    history = train_model(segmentation_model, train_seq, val_seq, training_callbacks)
    models.save_model(segmentation_model, str(artifacts_folder / 'seg_model_best.h5'))

    return history.history['val_loss'][-1]
