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
from models_sdd import SDDModel
from utils.data_utils import DataSequence

_CUDA_VISIBLE_DEVICES = "3"
_MODEL_ARC = 'deeplab'
os.environ["CUDA_VISIBLE_DEVICES"] = _CUDA_VISIBLE_DEVICES
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

ex = Experiment(name=_MODEL_ARC)
ex.observers.append(MongoObserver.create(
    url='127.0.0.1:27017', db_name='severstal_sdd'))
ex.captured_out_filter = apply_backspaces_and_linefeeds


# noinspection PyUnusedLocal
@ex.config
def config():
    seed = 42
    gpu_count = len(_CUDA_VISIBLE_DEVICES.split(','))
    backbone = 'resnet18'
    batch_size = 5
    lr = 0.0001
    dropout_rate = 0.2
    data_path = 'data'
    artifacts_folder = f'artifacts/{_MODEL_ARC}/{backbone}/{time.strftime("%d-%m-%y_%H:%M", time.localtime())}'
    img_size = (256, 1600, 3)
    use_multi_gpu = gpu_count > 1
    use_cbam = False
    do_classification = _MODEL_ARC == 'deeplab_classification_binary'
    optimizer = 'adam'
    folds = 2


@ex.automain
def run(backbone, batch_size, lr, dropout_rate, data_path, artifacts_folder, optimizer,
        img_size, use_multi_gpu, gpu_count, use_cbam, do_classification, folds, seed, _run):
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

    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=seed)
    print(f'length of train and val data: {len(train_df.index)}, {len(val_df.index)}')

    parallel_model = None
    score = 0
    for idx in range(folds):
        print(f'\n=====================================Training iteration: {idx}=====================================')
        scale = folds - idx
        temp_img_size = (img_size[0] // scale, img_size[1] // scale, img_size[2])
        temp_lr = lr / (10**idx)
        temp_batch_size = batch_size * scale + (batch_size - 2) * (scale - 1)
        print(f'image size: {temp_img_size}')
        print(f'scale: {scale}')
        print(f'learning rate: {temp_lr}')
        print(f'batch size: {temp_batch_size}\n')
        if use_multi_gpu:
            sdd_model, parallel_model = SDDModel(backbone, temp_img_size, temp_lr, dropout_rate, _MODEL_ARC,
                                                 use_multi_gpu=use_multi_gpu, gpu_count=gpu_count,
                                                 use_cbam=use_cbam, optimizer=optimizer).get_model()
        else:
            sdd_model = SDDModel(backbone, temp_img_size, temp_lr, dropout_rate, _MODEL_ARC, use_cbam=use_cbam,
                                 optimizer=optimizer).get_model()
        if idx > 0:
            print(f'loading weights from best_model of previous iteration...')
            sdd_model.load_weights(str(artifacts_folder / f'best_model_weights_{idx-1}.h5'))
        sdd_model.summary()
        utils.plot_model(sdd_model, str(artifacts_folder / f'model{idx}.png'), show_shapes=True)

        ckpt_path = artifacts_folder / 'ckpts' / f'model{idx}'
        ckpt_path.mkdir(exist_ok=True, parents=True)
        training_callbacks = [
            callbacks.ReduceLROnPlateau(patience=3, verbose=1),
            callbacks.EarlyStopping(patience=5, verbose=1, restore_best_weights=True),
            callbacks.ModelCheckpoint(str(ckpt_path / 'model-{epoch:04d}-{val_loss:.4f}.hdf5'),
                                      verbose=1, save_best_only=True),
            callbacks.TensorBoard(log_dir=str(artifacts_folder / 'tb_logs' / f'model{idx}')),
            callbacks.TerminateOnNaN(),
            ObserveMetrics(_run)
        ]

        train_seq = DataSequence(seed, train_df, temp_batch_size, temp_img_size, 'data/train_images',
                                 shuffle=True, augment=True, classification=do_classification)
        val_seq = DataSequence(seed, val_df, temp_batch_size, temp_img_size, 'data/train_images', shuffle=False,
                               augment=False, classification=do_classification)
        if use_multi_gpu:
            training_model = parallel_model
        else:
            training_model = sdd_model
        history = training_model.fit_generator(train_seq, epochs=100, verbose=2, callbacks=training_callbacks,
                                               validation_data=val_seq, max_queue_size=4, workers=4,
                                               use_multiprocessing=True)
        models.save_model(sdd_model, str(artifacts_folder / f'model_best{idx}.h5'))
        if idx != (folds - 1):
            print(f'Saving weights of best model of iteration {idx}')
            sdd_model.save_weights(str(artifacts_folder / f'best_model_weights_{idx}.h5'))

        pd.DataFrame(history.history).to_csv(f'history{idx}.csv', index=False)
        _run.add_artifact(f'history{idx}.csv')
        os.remove(f'history{idx}.csv')

        score += history.history['val_score'][-1]
    score /= folds
    return score
