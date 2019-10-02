import os
import time
import random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# import tensorflow as tf
from keras import callbacks, utils, models

from utils.data_utils import DataSequence
from models_sdd import SDDModel


def run(backbone, batch_size, lr, dropout_rate, data_path, artifacts_folder, img_size, use_multi_gpu):
    artifacts_folder = Path(artifacts_folder)
    artifacts_folder.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_path)
    data_df = pd.read_csv(data_path / 'train.csv')
    data_df['image_id'] = data_df.ImageId_ClassId.apply(lambda x: x.split('_')[0])
    data_df['class_id'] = data_df.ImageId_ClassId.apply(lambda x: x.split('_')[1])
    data_df.drop('ImageId_ClassId', axis=1, inplace=True)
    train_df = pd.DataFrame({
        'image_id': data_df['image_id'][::4]
    })
    train_df['defect_1'] = data_df.EncodedPixels[::4].values
    train_df['defect_2'] = data_df.EncodedPixels[1::4].values
    train_df['defect_3'] = data_df.EncodedPixels[2::4].values
    train_df['defect_4'] = data_df.EncodedPixels[3::4].values
    train_df['defect_count'] = train_df[train_df.columns[1:]].count(axis=1)
    train_df.reset_index(inplace=True, drop=True)
    train_df.fillna('', inplace=True)
    print(train_df.info())
    print(train_df.head(10))

    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=_SEED)
    print(f'length of train and val data: {len(train_df.index)}, {len(val_df.index)}')

    ckpt_path = artifacts_folder / 'ckpts'
    ckpt_path.mkdir(exist_ok=True, parents=True)
    parallel_model = None
    if use_multi_gpu:
        sdd_model, parallel_model = SDDModel(backbone, img_size, lr, dropout_rate, model_arc,
                                             use_multi_gpu=use_multi_gpu, gpu_count=gpu_count).get_model()
    else:
        sdd_model = SDDModel(backbone, img_size, lr, dropout_rate).get_model()
    sdd_model.summary()
    utils.plot_model(sdd_model, str(artifacts_folder / 'model.png'), show_shapes=True)
    training_callbacks = [
        callbacks.ReduceLROnPlateau(patience=3, verbose=1),
        callbacks.EarlyStopping(patience=5, verbose=1, restore_best_weights=True),
        callbacks.ModelCheckpoint(str(ckpt_path / 'model-{epoch:04d}-{val_loss:.4f}.hdf5'),
                                  verbose=1, save_best_only=True),
        callbacks.TensorBoard(log_dir=str(artifacts_folder / 'tb_logs')),
        callbacks.TerminateOnNaN()
    ]

    train_seq = DataSequence(_SEED, train_df, batch_size, img_size, 'data/train_images', shuffle=True, augment=True)
    val_seq = DataSequence(_SEED, val_df, batch_size, img_size, 'data/train_images', shuffle=False, augment=False)
    if use_multi_gpu:
        training_model = parallel_model
    else:
        training_model = sdd_model
    history = training_model.fit_generator(train_seq, epochs=100, verbose=1, callbacks=training_callbacks,
                                           validation_data=val_seq, max_queue_size=4, workers=4,
                                           use_multiprocessing=True)
    models.save_model(sdd_model, str(artifacts_folder / 'model_best.h5'))

    pd.DataFrame(history.history).to_csv(artifacts_folder / 'history.csv', index=False)


if __name__ == '__main__':
    start_time = time.time()
    cuda_visible_devices = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    _SEED = 42
    random.seed(_SEED)
    np.random.seed(_SEED)
    # tf.compat.v1.random.set_random_seed(_SEED)

    gpu_count = len(cuda_visible_devices.split(','))
    model_arc = 'deeplab'
    params = {
        'backbone': 'resnet18',
        'batch_size': 4,
        'lr': 0.0001,
        'dropout_rate': 0.2,
        'data_path': 'data',
        'artifacts_folder': f'artifacts/{model_arc}/{time.strftime("%d-%m-%y_%H:%M", time.localtime())}',
        'img_size': (256, 1600, 3),
        'use_multi_gpu': gpu_count > 1
    }
    run(**params)
    print(f'Total run time: {(time.time() - start_time) / 60} minutes.')
