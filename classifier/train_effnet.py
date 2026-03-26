import os
import argparse
import numpy as np
import torch
import warnings
from torch import DoubleTensor
from torch.utils.data import DataLoader
from torch.utils.data.sampler import (SubsetRandomSampler,
                                      WeightedRandomSampler)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
try:
    from pytorch_lightning.loggers import NeptuneLogger
except ImportError:
    from pytorch_lightning.loggers.neptune import NeptuneLogger
from efficientnet_pytorch import EfficientNet
from torchvision import datasets
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from models.efficientnet import LitterClassification

from train_resnet import make_weights_for_balanced_classes


CLASS_TO_SUPERCLASS = {
    # paper
    'leaflet': 'paper',
    'newspaper': 'paper',
    'napkin': 'paper',
    # plastic
    'plasticbag': 'plastic',
    'plasticbottle': 'plastic',
    'plasticene': 'plastic',
    # metal
    'cans': 'metal',
    'battery': 'metal',
    # glass
    'glassbottle': 'glass',
    'bulb': 'glass',
    # food
    'bread': 'food',
    'leftovers': 'food',
    'watermelonrind': 'food',
    'nut': 'food',
    # medical
    'bandaid': 'medical',
    'diapers': 'medical',
    'facialmask': 'medical',
    'medicinebottle': 'medical',
    'tabletcapsule': 'medical',
    'thermometer': 'medical',
    'traditionalChinesemedicine': 'medical',
    # personal care
    'toothbrush': 'personal_care',
    'toothpastetube': 'personal_care',
    'nailpolishbottle': 'personal_care',
    # explicit "other" examples
    'bowlsanddishes': 'other',
    'chopsticks': 'other',
    'penholder': 'other',
    'pesticidebottle': 'other',
    'toothpick': 'other',
    'XLight': 'other',
}

SUPER_CLASSES = [
    'paper',
    'plastic',
    'metal',
    'glass',
    'food',
    'medical',
    'personal_care',
    'other',
]


class SafeImageFolder(datasets.ImageFolder):
    """ImageFolder variant that skips unreadable/corrupted image files."""

    def __init__(self, *args, max_skip_retries=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_skip_retries = max_skip_retries

    def __getitem__(self, index):
        # If an image is corrupted, move to the next sample and continue.
        for _ in range(self.max_skip_retries):
            try:
                return super().__getitem__(index)
            except Exception as err:
                sample_path = self.samples[index][0]
                warnings.warn(
                    f"Skipping unreadable image: {sample_path} ({err})",
                    RuntimeWarning,
                )
                index = (index + 1) % len(self.samples)

        raise RuntimeError(
            "Too many unreadable images in a row. "
            "Please clean the dataset files."
        )


class RemappedImageFolder(torch.utils.data.Dataset):
    """Wrap ImageFolder and remap original classes to 8 super-classes."""

    def __init__(self, base_dataset, class_to_superclass, super_classes):
        self.base = base_dataset
        self.super_classes = list(super_classes)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.super_classes)}
        self.classes = list(self.super_classes)

        idx_to_class = {v: k for k, v in self.base.class_to_idx.items()}
        self.samples = []
        self.imgs = []
        self.targets = []

        for path, old_idx in self.base.samples:
            old_class_name = idx_to_class[old_idx]
            super_class = class_to_superclass.get(old_class_name, 'other')
            new_idx = self.class_to_idx[super_class]
            self.samples.append((path, new_idx))
            self.imgs.append((path, new_idx))
            self.targets.append(new_idx)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, _ = self.base[index]
        return image, self.targets[index]


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Train efficientnet')

    # Dataset / Model parameters
    parser.add_argument(
        '--data', metavar='DIR',
        help='path to base directory with data',
        default='/dih4/dih4_2/wimlds/amikolajczyk/detect-waste/classifier/')
    parser.add_argument(
        '--save', metavar='OUTPUT',
        help='path to directory to save checkpoint',
        default='/dih4/dih4_2/wimlds/smajchrowska/classifier/effnet.ckpt')
    parser.add_argument(
        '--model', default='efficientnet-b0', type=str,
        help='Name of model to train (default: "efficientnet-b0)"')
    parser.add_argument(
        '--lr', type=float, default=0.0001,
        help='learning rate (default: 0.0001)')
    parser.add_argument(
        '--decay', type=float, default=0.99,
        help='learning rate (default: 0.99)')
    parser.add_argument(
        '-b', '--batch-size', type=int, default=16,
        help='input batch size for training (default: 16)')
    parser.add_argument(
        '--epochs', type=int, default=20, metavar='EPOCHS',
        help='number of epochs to train (default: 20)')
    parser.add_argument(
        '--num-classes', type=int, default=None, metavar='NUM',
        help='number of classes to classify (default: infer from dataset)')
    parser.add_argument(
        '--gpu', type=int, default=7, metavar='GPU',
        help='GPU number to use (default: 7)')
    parser.add_argument(
        '--weighted_sampler', action='store_true', default=False,
        help="for unbalanced dataset you can create a weighted sampler"
             "(default: False)")
    parser.add_argument('--pseudolabel_mode',
                        help='type actualization of pseudolabeling',
                        default='per_epoch',
                        choices=['per_batch', 'per_epoch'],
                        type=str)
    parser.add_argument(
        '--neptune', action='store_true', default=False,
        help="enable neptune launch")
    parser.add_argument(
        '--superclass-map', type=str, default='none',
        choices=['none', '8class'],
        help='Optional label remapping. Use "8class" to map old folders into '
             'paper/plastic/metal/glass/food/medical/personal_care/other.')
    parser.set_defaults(redundant_bias=None)
    return parser


def make_sampler(split_set, weighted_sampler=False):
    indices = list(range(len(split_set)))
    if weighted_sampler:
        # For unbalanced dataset we create a weighted sampler
        sampler = []
        for i in indices:
            sampler.append(split_set.imgs[i])
        weights = make_weights_for_balanced_classes(sampler,
                                                    len(split_set.classes))
        sampler = WeightedRandomSampler(DoubleTensor(weights), len(weights))
    else:
        sampler = SubsetRandomSampler(indices)
    return sampler


def get_augmentation(transform):
    return lambda img: transform(image=np.array(img))


def main(args):
    TRAIN_DIR = os.path.join(args.data, 'images_square', 'train')
    TEST_DIR = os.path.join(args.data, 'images_square', 'test')
    PSEUDO_DIR = os.path.join(args.data, 'images_square', 'pseudolabel')
    img_size = EfficientNet.get_image_size(args.model)
    train_transform = A.Compose([A.Resize(img_size + 60, img_size + 60),
                                 A.RandomCrop(img_size, img_size),
                                 A.HorizontalFlip(),
                                 A.VerticalFlip(),
                                 A.ShiftScaleRotate(),
                                 A.RandomBrightnessContrast(),
                                 A.CoarseDropout(
                                     num_holes_range=(1, 8),
                                     hole_height_range=(0.05, 0.2),
                                     hole_width_range=(0.05, 0.2),
                                     fill=0,
                                 ),
                                 A.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
                                 ToTensorV2()])
    test_transform = A.Compose([A.Resize(img_size, img_size),
                                A.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                                ToTensorV2()])

    train_set = SafeImageFolder(
        root=TRAIN_DIR,
        transform=get_augmentation(train_transform))

    test_set = SafeImageFolder(
        root=TEST_DIR,
        transform=get_augmentation(test_transform))

    if args.superclass_map == '8class':
        train_set = RemappedImageFolder(train_set, CLASS_TO_SUPERCLASS, SUPER_CLASSES)
        test_set = RemappedImageFolder(test_set, CLASS_TO_SUPERCLASS, SUPER_CLASSES)

    dataset_num_classes = len(train_set.classes)
    if args.num_classes is None:
        num_classes = dataset_num_classes
    else:
        num_classes = args.num_classes
        if num_classes != dataset_num_classes:
            raise ValueError(
                f"--num-classes={num_classes} does not match dataset "
                f"classes={dataset_num_classes} in {TRAIN_DIR}."
            )

    # add weighted or random sampler
    train_sampler = make_sampler(train_set, args.weighted_sampler)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              sampler=train_sampler,
                              num_workers=args.batch_size)
    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.batch_size)

    if os.path.isdir(PSEUDO_DIR):
        pseudo_set = SafeImageFolder(
            root=PSEUDO_DIR,
            transform=get_augmentation(train_transform))
        if args.superclass_map == '8class':
            pseudo_set = RemappedImageFolder(
                pseudo_set,
                CLASS_TO_SUPERCLASS,
                SUPER_CLASSES,
            )
        pseudo_loader = DataLoader(pseudo_set,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.batch_size)

        model = LitterClassification(model_name=args.model,
                                     lr=args.lr,
                                     decay=args.decay,
                                     num_classes=num_classes,
                                     pseudoloader=pseudo_loader,
                                     pseudolabel_mode=args.pseudolabel_mode,
                                     class_to_idx=train_set.class_to_idx,
                                     classes=train_set.classes)
    else:
        model = LitterClassification(model_name=args.model,
                                     lr=args.lr,
                                     decay=args.decay,
                                     num_classes=num_classes,
                                     class_to_idx=train_set.class_to_idx,
                                     classes=train_set.classes)

    ckpt_dir = os.path.dirname(args.save) if args.save else None
    best_checkpoint = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        verbose=True,
        filename='best-{epoch:02d}-{val_loss:.4f}',
    )
    last_checkpoint = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_last=True,
        save_top_k=0,
        every_n_epochs=2,
        verbose=True,
        filename='last-{epoch:02d}',
    )

    if args.neptune:
        # your NEPTUNE_API_TOKEN should be add to ~./bashrc to run this file
        logger = NeptuneLogger(project='detectwaste/classification',
                               tags=[args.model, TRAIN_DIR])
    else:
        logger = None

    # Lightning 2.x uses accelerator/devices instead of gpus
    use_cuda = torch.cuda.is_available()
    trainer = pl.Trainer(accelerator='gpu' if use_cuda else 'cpu',
                         devices=[args.gpu] if use_cuda else 1,
                         max_epochs=args.epochs,
                         callbacks=[best_checkpoint, last_checkpoint],
                         logger=logger)
    trainer.fit(model, train_loader, test_loader)

    # manually you can save best checkpoints
    if args.save:
        trainer.save_checkpoint(args.save)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
