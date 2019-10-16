import os
import time
from pathlib import Path

import pandas as pd
from keras import callbacks, utils, models
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from sklearn.model_selection import train_test_split

from custom_objects.callbacks import ObserveMetrics
from models_sdd import SegmentationModel, ClassificationModel
from utils.data_utils import DataSequence, prepare_data_df, ClassificationDataSeq

_CUDA_VISIBLE_DEVICES = "3"
_MODEL_ARC = 'deeplab'
os.environ["CUDA_VISIBLE_DEVICES"] = _CUDA_VISIBLE_DEVICES
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
    seed = 42
    gpu_count = len(_CUDA_VISIBLE_DEVICES.split(','))
    backbone = 'resnet18'
    cfn_backbone = 'resnet18'
    batch_size = 4
    lr = 0.0001
    dropout_rate = 0.2
    data_path = 'data'
    artifacts_folder = f'artifacts/{_MODEL_ARC}/{backbone}/{time.strftime("%d-%m-%y_%H:%M", time.localtime())}'
    img_size = (256, 1600, 3)
    use_multi_gpu = gpu_count > 1
    use_cbam = False
    use_se = False
    cfn_model_path = None
    use_transpose_conv = False


@ex.automain
def run(backbone, cfn_backbone, batch_size, lr, dropout_rate, data_path, artifacts_folder,
        img_size, use_cbam, use_se, cfn_model_path, use_transpose_conv, seed, _run):
    artifacts_folder = Path(artifacts_folder)
    artifacts_folder.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_path)
    data_df = pd.read_csv(data_path / 'train.csv')
    data_df = prepare_data_df(data_df)
    print(data_df.info())
    print(data_df.head(10))

    train_df, val_df = train_test_split(data_df, test_size=0.2, random_state=seed)
    print(f'length of train and val data: {len(train_df.index)}, {len(val_df.index)}')

    ckpt_path = artifacts_folder / 'ckpts'
    ckpt_path.mkdir(exist_ok=True, parents=True)

    if cfn_model_path is None:
        classification_model = ClassificationModel(cfn_backbone, img_size, lr).get_model()
        utils.plot_model(classification_model, str(artifacts_folder / 'cfn_model.png'), show_shapes=True)
        training_callbacks = [
            callbacks.ReduceLROnPlateau(patience=3, verbose=1, min_lr=1e-7),
            callbacks.EarlyStopping(patience=5, verbose=1, restore_best_weights=True),
            callbacks.ModelCheckpoint(str(ckpt_path / 'cfn_model-{epoch:04d}-{val_loss:.4f}.hdf5'),
                                      verbose=1, save_best_only=True),
            callbacks.TensorBoard(log_dir=str(artifacts_folder / 'tb_logs')),
            callbacks.TerminateOnNaN(),
            ObserveMetrics(_run, 'cfn')
        ]
        train_seq = ClassificationDataSeq(seed, train_df, batch_size*4, img_size, 'data/train_images', mode='train',
                                          shuffle=True, augment=True)
        val_seq = ClassificationDataSeq(seed, val_df, batch_size*4, img_size, 'data/train_images', mode='val',
                                        shuffle=False, augment=False)
        train_model(classification_model, train_seq, val_seq, training_callbacks)
        models.save_model(classification_model, str(artifacts_folder / 'cfn_model_best.h5'))
    else:
        classification_model = models.load_model(cfn_model_path, compile=False)

    segmentation_model = SegmentationModel(backbone, img_size, lr, dropout_rate, _MODEL_ARC,
                                           use_cbam=use_cbam, use_se=use_se,
                                           cfn_model=classification_model,
                                           cfn_backbone=cfn_backbone,
                                           use_transpose_conv=use_transpose_conv).get_model()
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
                             augment=True)
    val_seq = DataSequence(seed, val_df, batch_size, img_size, 'data/train_images', mode='val', shuffle=False,
                           augment=False)

    history = train_model(segmentation_model, train_seq, val_seq, training_callbacks)
    models.save_model(segmentation_model, str(artifacts_folder / 'seg_model_best.h5'))

    return history.history['val_loss'][-1]
