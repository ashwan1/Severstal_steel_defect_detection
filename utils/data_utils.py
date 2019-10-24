import numpy as np
import cv2
import random
import pandas as pd
from tqdm import tqdm
from keras.utils import Sequence

from utils.rle_utils import rle2mask


class DataSequence(Sequence):
    def __init__(self, seed, df, batch_size, img_size,
                 base_path, mode='train', n_classes=4, shuffle=False,
                 augment=False, for_stacker=False):
        self.seed = seed
        self.df = df
        self.batch_size = batch_size
        self.height, self.width, self.n_channels = img_size
        self.base_path = base_path
        self.mode = mode
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.for_stacker = for_stacker

    def __len__(self):
        return int(np.ceil(len(self.df.index) / float(self.batch_size)))

    def __getitem__(self, idx):
        flip_direction = None
        masks = None
        batch = self.df[idx * self.batch_size: (idx + 1) * self.batch_size].reset_index(drop=True)
        images = np.zeros((len(batch.index), self.height, self.width, self.n_channels))
        if self.mode != 'test':
            masks = np.zeros((len(batch.index), self.height, self.width, self.n_classes), dtype='int')
        for row in batch.itertuples():
            image = cv2.imread(f'{self.base_path}/{row.image_id}')
            assert image.shape == (self.height, self.width, self.n_channels), \
                f'Image shape not as expected, got {image.shape}, expected {(self.height, self.width, self.n_channels)}'
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.augment:
                flip_direction = random.choice([-1, 0, 1])
                image = self.augment_image(image, row.defect_count, flip_direction=flip_direction)
            images[row.Index] = image / 255.
            if self.mode != 'test':
                rles = row[2:6]
                mask = self.build_mask(rles, flip_direction)
                masks[row.Index] = mask
        if self.mode != 'test':
            if self.for_stacker:
                return [images, images], masks
            else:
                return images, masks
        else:
            if self.for_stacker:
                return [images, images]
            else:
                return images

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

    def augment_image(self, image, defect_count, flip_direction):
        image = cv2.flip(image, flip_direction)
        if random.random() > 0.5:
            alpha = random.uniform(1.0, 2.0)  # Simple contrast control
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        elif random.random() > 0.6:
            beta = random.randint(10, 61)  # Simple brightness control
            image = cv2.convertScaleAbs(image, alpha=1, beta=beta)
        elif random.random() > 0.7:
            image = self.illuminate(image)
        if random.random() > 0.7:
            image = self.add_noise(image, 'salt_pepper')
        elif random.random() > 0.5:
            image = self.add_noise(image, 'gauss')
        if defect_count == 0:
            if random.random() > 0.5:
                crop_hw = [random.choice(range(10, self.height // 10)), random.choice(range(10, self.width // 10))]
                image = image[crop_hw[0]:-crop_hw[0], crop_hw[1]:-crop_hw[1], :]
                image = cv2.resize(image, (self.width, self.height))
            if random.random() > 0.5:
                transformation_matrix = np.float32([[1, 0, random.choice(range(-self.width // 4, self.width // 4))],
                                                    [0, 1,
                                                     random.choice(range(-self.height // 10, self.height // 10))]])
                image = cv2.warpAffine(image, transformation_matrix, (self.width, self.height))
        return image

    def build_mask(self, rles, flip_direction=None):
        assert self.n_classes == len(rles), 'length of rles should be same as number of classes'
        mask = np.zeros((self.height, self.width, self.n_classes), dtype='int')
        for i, rle in enumerate(rles):
            if type(rle) is str:
                m = rle2mask(rle, (self.width, self.height))
                if flip_direction is not None:
                    mask[:, :, i] = cv2.flip(m, flip_direction)
                else:
                    mask[:, :, i] = m
        return mask

    def illuminate(self, image):
        illumination = np.zeros((self.height, self.width), np.float32)
        cv2.circle(illumination, (random.choice(range(0, self.width)), random.choice(range(0, self.height))),
                   random.choice(range(10, 150)), 1, -1, lineType=cv2.LINE_AA)
        illumination = cv2.GaussianBlur(illumination, (257, 257), 0)
        illumination = illumination.reshape(self.height, self.width, 1)
        image = image.astype(np.float32) / 255
        image = image * (1 + illumination * 1.05)
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        return image

    def add_noise(self, image, noise_type):
        image = image / 255.
        if noise_type == 'salt_pepper':
            s_vs_p = 0.5
            amount = 0.004
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            image[tuple(coords)] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            image[tuple(coords)] = 0
        elif noise_type == 'gauss':
            mean = 0
            var = 0.1
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (self.height, self.width, self.n_channels))
            image = image + gauss

        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        return image


class ClassificationDataSeq(Sequence):
    def __init__(self, seed, df, batch_size, img_size,
                 base_path, mode='train', n_classes=4, shuffle=False,
                 augment=False):
        self.seed = seed
        self.df = df
        self.batch_size = batch_size
        self.height, self.width, self.n_channels = img_size
        self.base_path = base_path
        self.mode = mode
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment

    def __len__(self):
        return int(np.ceil(len(self.df.index) / float(self.batch_size)))

    def __getitem__(self, idx):
        labels = None
        batch = self.df[idx * self.batch_size: (idx + 1) * self.batch_size].reset_index(drop=True)
        images = np.zeros((len(batch.index), self.height, self.width, self.n_channels))
        if self.mode != 'test':
            labels = np.zeros((len(batch.index), self.n_classes))
        for row in batch.itertuples():
            image = cv2.imread(f'{self.base_path}/{row.image_id}')
            assert image.shape == (self.height, self.width, self.n_channels), \
                f'Image shape not as expected, got {image.shape}, expected {(self.height, self.width, self.n_channels)}'
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.augment:
                image = self.augment_image(image)
            images[row.Index] = image / 255.
            if self.mode != 'test':
                labels[row.Index] = row[7:]
        if self.mode != 'test':
            return images, labels
        else:
            return images

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

    def augment_image(self, image):
        flip_direction = random.choice([-1, 0, 1])
        image = cv2.flip(image, flip_direction)
        if random.random() > 0.5:
            alpha = random.uniform(1.0, 2.0)  # Simple contrast control
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        elif random.random() > 0.6:
            beta = random.randint(10, 61)  # Simple brightness control
            image = cv2.convertScaleAbs(image, alpha=1, beta=beta)
        elif random.random() > 0.7:
            image = self.illuminate(image)
        if random.random() > 0.7:
            image = self.add_noise(image, 'salt_pepper')
        elif random.random() > 0.5:
            image = self.add_noise(image, 'gauss')
        if random.random() > 0.5:
            crop_hw = [random.choice(range(10, self.height // 10)), random.choice(range(10, self.width // 10))]
            image = image[crop_hw[0]:-crop_hw[0], crop_hw[1]:-crop_hw[1], :]
            image = cv2.resize(image, (self.width, self.height))
        if random.random() > 0.5:
            transformation_matrix = np.float32([[1, 0, random.choice(range(-self.width // 4, self.width // 4))],
                                                [0, 1,
                                                 random.choice(range(-self.height // 10, self.height // 10))]])
            image = cv2.warpAffine(image, transformation_matrix, (self.width, self.height))
        return image

    def illuminate(self, image):
        illumination = np.zeros((self.height, self.width), np.float32)
        cv2.circle(illumination, (random.choice(range(0, self.width)), random.choice(range(0, self.height))),
                   random.choice(range(10, 150)), 1, -1, lineType=cv2.LINE_AA)
        illumination = cv2.GaussianBlur(illumination, (257, 257), 0)
        illumination = illumination.reshape(self.height, self.width, 1)
        image = image.astype(np.float32) / 255
        image = image * (1 + illumination * 1.05)
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        return image

    def add_noise(self, image, noise_type):
        image = image / 255.
        if noise_type == 'salt_pepper':
            s_vs_p = 0.5
            amount = 0.004
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            image[tuple(coords)] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            image[tuple(coords)] = 0
        elif noise_type == 'gauss':
            mean = 0
            var = 0.1
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (self.height, self.width, self.n_channels))
            image = image + gauss

        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        return image


def prepare_data_df(data_df):
    data_df['image_id'] = data_df.ImageId_ClassId.apply(lambda x: x.split('_')[0])
    data_df['class_id'] = data_df.ImageId_ClassId.apply(lambda x: x.split('_')[1])
    data_df.drop('ImageId_ClassId', axis=1, inplace=True)
    df = pd.DataFrame({
        'image_id': data_df['image_id'][::4]
    })
    df['defect_1'] = data_df.EncodedPixels[::4].values
    df['defect_2'] = data_df.EncodedPixels[1::4].values
    df['defect_3'] = data_df.EncodedPixels[2::4].values
    df['defect_4'] = data_df.EncodedPixels[3::4].values
    df['defect_count'] = df[df.columns[1:]].count(axis=1)
    df.reset_index(inplace=True, drop=True)
    df['has_defect_1'] = (~df['defect_1'].isna()).astype(np.int8)
    df['has_defect_2'] = (~df['defect_2'].isna()).astype(np.int8)
    df['has_defect_3'] = (~df['defect_3'].isna()).astype(np.int8)
    df['has_defect_4'] = (~df['defect_4'].isna()).astype(np.int8)
    df.fillna('', inplace=True)
    return df


def generate_mix_match(orig_df, defect_type, count):
    mix_match_dict = {
        'image_id': [],
        'defect_1': [],
        'defect_2': [],
        'defect_3': [],
        'defect_4': [],
        'defect_count': [],
        'has_defect_1': [],
        'has_defect_2': [],
        'has_defect_3': [],
        'has_defect_4': []
    }
    no_defect_df = orig_df[orig_df.defect_count == 0]
    defect_df = orig_df[~orig_df[f'defect_{defect_type}'].isna() & orig_df.defect_count == 1]
    for _ in tqdm(range(count), desc=f'generating {count} mix-matches for defect {defect_type}'):
        no_defect_img_name = no_defect_df.sample().iloc[0].image_id
        no_defect_img = _read_img(f'data/train_images/{no_defect_img_name}')
        defective_sample = defect_df.sample()
        defect_img = _read_img(f'data/train_images/{defective_sample.iloc[0].image_id}')
        mask = rle2mask(defective_sample.iloc[0][f'defect_{defect_type}'])
        contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            no_defect_img[y:y + h, x:x + w, :] = defect_img[y:y + h, x:x + w, :]
        _write_image(f'data/train_images/aug_{defect_type}_{no_defect_img_name}', no_defect_img)
        mix_match_dict['image_id'].append(f'aug_{defect_type}_{no_defect_img_name}')
        mix_match_dict[f'defect_{defect_type}'].append(defective_sample.iloc[0][f'defect_{defect_type}'])
        mix_match_dict['defect_count'].append(1)
        mix_match_dict[f'has_defect_{defect_type}'].append(1)
        for j in range(1, 5):
            if j != defect_type:
                mix_match_dict[f'defect_{j}'].append(np.nan)
                mix_match_dict[f'has_defect_{j}'].append(0)
    mix_match_df = pd.DataFrame(mix_match_dict)
    return mix_match_df


def _read_img(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError
    return img


def _write_image(path, img):
    result = cv2.imwrite(path, img)
    if not result:
        raise FileNotFoundError


def duplicate_data(orig_data, defect_class, times):
    new_data = orig_data.copy()
    for _ in tqdm(range(times), desc=f'duplicating defect {defect_class}'):
        new_data = new_data.append(
            orig_data[orig_data[f'has_defect_{defect_class}'] == 1][orig_data.defect_count == 1])
    new_data = new_data.sample(frac=1).reset_index(drop=True)
    return new_data
