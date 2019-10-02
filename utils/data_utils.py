import numpy as np
import cv2
import random
from keras.utils import Sequence

from utils.rle_utils import rle2mask


class DataSequence(Sequence):
    def __init__(self, seed, df, batch_size, img_size,
                 base_path, train=True, n_classes=4, shuffle=False, augment=False):
        self.seed = seed
        self.df = df
        self.batch_size = batch_size
        self.height, self.width, self.n_channels = img_size
        self.base_path = base_path
        self.train = train
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment

    def __len__(self):
        return int(np.ceil(len(self.df.index) / float(self.batch_size)))

    def __getitem__(self, idx):
        flip_direction = None
        masks = None
        batch = self.df[idx * self.batch_size: (idx + 1) * self.batch_size].reset_index(drop=True)
        images = np.zeros((len(batch.index), self.height, self.width, self.n_channels))
        if self.train:
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
            if self.train:
                rles = row[2:-1]
                mask = self.build_mask(rles, flip_direction)
                masks[row.Index] = mask
        if self.train:
            return images, masks
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
        if defect_count == 0:
            if random.random() > 0.7:
                image = self.illuminate(image)
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
