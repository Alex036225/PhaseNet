"""PhaseNet training / testing entry."""

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import get_config
from dataset.data_loader.BaseLoader import BaseLoader
from dataset.data_loader.MMPDLoader import MMPDLoader
from dataset.data_loader.UBFCrPPGLoader import UBFCrPPGLoader
from dataset.data_loader.ZhuhaiLoader import ZhuhaiLoader
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.trainer.PhaseNetTrainer import PhaseNetTrainer


RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)
train_generator = torch.Generator()
train_generator.manual_seed(RANDOM_SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def add_args(parser):
    parser.add_argument(
        "--config_file",
        required=False,
        default="configs/PhaseNet-UUU.yaml",
        type=str,
        help="Path to the PhaseNet yaml config.",
    )
    return parser


def resolve_loader(dataset_name):
    mapping = {
        "UBFC-rPPG": UBFCrPPGLoader,
        "MMPD": MMPDLoader,
        "Zhuhai": ZhuhaiLoader,
    }
    if dataset_name not in mapping:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            "This bundle supports only UBFC-rPPG, MMPD, and Zhuhai."
        )
    return mapping[dataset_name]


def build_loader(name, loader_cls, data_path, config_data, batch_size, shuffle, generator, drop_last=False):
    dataset = loader_cls(name=name, data_path=data_path, config_data=config_data)
    return DataLoader(
        dataset=dataset,
        num_workers=16,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_data_loaders(config):
    data_loader_dict = {}

    if config.TOOLBOX_MODE == "multi_train_and_test":
        for i in range(1, config.TRAIN.DATA.MULTI_SOURCE.NUM_SOURCE + 1):
            source_name = config.TRAIN.DATA.MULTI_SOURCE.SOURCE_NAME[i - 1]
            loader_cls = resolve_loader(source_name)
            key = f"train{i}"
            data_loader_dict[key] = build_loader(
                name=key,
                loader_cls=loader_cls,
                data_path=config.TRAIN.DATA.DATA_PATH,
                config_data=config.TRAIN.DATA,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=True,
                generator=train_generator,
                drop_last=True,
            )

        if config.VALID.DATA.DATASET and config.VALID.DATA.DATA_PATH and not config.TEST.USE_LAST_EPOCH:
            valid_cls = resolve_loader(config.VALID.DATA.DATASET)
            data_loader_dict["valid"] = build_loader(
                name="valid",
                loader_cls=valid_cls,
                data_path=config.VALID.DATA.DATA_PATH,
                config_data=config.VALID.DATA,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=False,
                generator=general_generator,
            )
        else:
            data_loader_dict["valid"] = None

    elif config.TOOLBOX_MODE == "train_and_test":
        train_cls = resolve_loader(config.TRAIN.DATA.DATASET)
        data_loader_dict["train"] = build_loader(
            name="train",
            loader_cls=train_cls,
            data_path=config.TRAIN.DATA.DATA_PATH,
            config_data=config.TRAIN.DATA,
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=True,
            generator=train_generator,
        )

        if config.VALID.DATA.DATASET and config.VALID.DATA.DATA_PATH and not config.TEST.USE_LAST_EPOCH:
            valid_cls = resolve_loader(config.VALID.DATA.DATASET)
            data_loader_dict["valid"] = build_loader(
                name="valid",
                loader_cls=valid_cls,
                data_path=config.VALID.DATA.DATA_PATH,
                config_data=config.VALID.DATA,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=False,
                generator=general_generator,
            )
        else:
            data_loader_dict["valid"] = None

    elif config.TOOLBOX_MODE != "only_test":
        raise ValueError("PhaseNet bundle supports only train_and_test, multi_train_and_test, and only_test.")

    if config.TOOLBOX_MODE in {"train_and_test", "multi_train_and_test", "only_test"}:
        test_cls = resolve_loader(config.TEST.DATA.DATASET)
        data_loader_dict["test"] = build_loader(
            name="test",
            loader_cls=test_cls,
            data_path=config.TEST.DATA.DATA_PATH,
            config_data=config.TEST.DATA,
            batch_size=config.INFERENCE.BATCH_SIZE,
            shuffle=False,
            generator=general_generator,
        )

    return data_loader_dict


def main():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = BaseTrainer.add_trainer_args(parser)
    parser = BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    config = get_config(args)
    if config.MODEL.NAME != "PhaseNet":
        raise ValueError(f"MODEL.NAME must be 'PhaseNet', got {config.MODEL.NAME!r}")

    data_loader_dict = build_data_loaders(config)
    model_trainer = PhaseNetTrainer(config, data_loader_dict)

    if config.TOOLBOX_MODE in {"train_and_test", "multi_train_and_test"}:
        model_trainer.train(data_loader_dict)
        model_trainer.test(data_loader_dict)
    elif config.TOOLBOX_MODE == "only_test":
        model_trainer.test(data_loader_dict)
    else:
        raise ValueError("Unsupported TOOLBOX_MODE for PhaseNet bundle.")


if __name__ == "__main__":
    main()

