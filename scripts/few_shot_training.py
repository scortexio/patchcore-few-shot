"""Anomalib Training Script.

This script reads the name of the model or config file from command
line, train/test the anomaly model to get quantitative and qualitative
results.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import gc
import logging
import os
import random
import shutil
import tempfile
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Union

import albumentations as A
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import Trainer, seed_everything

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.data.augmentation import Transformer
from anomalib.data.utils.download import download_and_extract
from anomalib.models import get_model
from anomalib.utils.callbacks import ModelCheckpoint, get_callbacks
from anomalib.utils.callbacks.visualizer import ImageVisualizerCallback, MetricVisualizerCallback
from anomalib.utils.loggers import configure_logger, get_experiment_logger
from anomalib.utils.metrics.saver import create_and_save_boxplots

logger = logging.getLogger("anomalib")
HOME_DIR = os.path.expanduser("~")


def train_step(
    config: Union[DictConfig, ListConfig],
    number_transforms: Union[int, None],
    augmentation_transforms: Union[A.Compose, None],
    output_metrics: list,
    k_shot: int,
    run: int,
    category: str,
):
    if bool(number_transforms) is not bool(augmentation_transforms):
        raise ValueError(
            f"number_transforms={number_transforms} and augmentation_transforms={augmentation_transforms}, "
            f"both variables must both None or both not None together."
        )

    # create path to save images and predictions as well as metric curves
    image_save_path = config.project.path + f"/images/{k_shot}shot/{run}/{category}/"
    config.visualization.image_save_path = image_save_path
    # re-set category
    config.dataset.category = category
    logger.info(f"Doing category: {config.dataset.category}")

    if config.dataset.name == "visa":
        config.dataset.format = "visa"
    elif config.dataset.name == "mvtec":
        config.dataset.format = "mvtec"
    else:
        raise ValueError(f"Dataset {config.dataset.name} not supported")

    datamodule = get_datamodule(config)
    # download dataset if it doesn't exist
    datamodule.prepare_data()
    datamodule.setup()

    sampling_indexes = random.sample(range(len(datamodule.train_data)), k=k_shot)

    datamodule.train_data.subsample(sampling_indexes, inplace=True)

    if config.project.augment:
        assert number_transforms is not None
        directory_augmented_images = tempfile.mkdtemp()
        datamodule.train_data.augment_train_set(
            directory_augmented_images, augmentation_transforms, number_transforms=number_transforms
        )
        assert len(datamodule.train_data) == len(sampling_indexes) * (number_transforms + 1)

    logger.info(f"Training on {len(datamodule.train_data)} examples")
    logger.info(datamodule.train_data._samples)

    model = get_model(config)
    experiment_logger = get_experiment_logger(config)
    callbacks = get_callbacks(config)
    # remove image saving callback, since we don't have ground-truths during submission
    callbacks_to_remove = [ImageVisualizerCallback, MetricVisualizerCallback, ModelCheckpoint]
    callbacks = [
        callback
        for callback in callbacks
        if not any(isinstance(callback, cb_remove) for cb_remove in callbacks_to_remove)
    ]

    # force disable intermediate checkpointing
    config.trainer.enable_checkpointing = False
    trainer = Trainer(**config.trainer, logger=experiment_logger, callbacks=callbacks)
    logger.info("Training the model.")
    trainer.fit(model=model, datamodule=datamodule)

    # metrics is a list of dictionaries
    metrics = {k: v.item() for k, v in trainer.logged_metrics.items()}

    # save metrics and weights information
    metrics_and_model_dict = {
        "k_shot": k_shot,
        "run": run,
        "category": category,
    }
    metrics_and_model_dict.update(metrics)
    output_metrics.append(metrics_and_model_dict)

    # remove temporary file with augmented images
    if config.project.augment:
        shutil.rmtree(directory_augmented_images)

    # save an intermediate result
    pd.DataFrame(output_metrics).to_csv(Path(config.project.path) / "metrics_and_weights.csv")


def get_args(raw_args=None) -> Namespace:
    """Get command line arguments.

    Returns:
        Namespace: List of arguments.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="patchcore",
        help="Name of the algorithm to train/test",
    )
    parser.add_argument("--dataset", type=str, default="visa", help="Dataset to do the training on")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to a model config file",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="wide_resnet50_2",
        help="The backbone of the AD algorithm/model",
    )
    parser.add_argument(
        "--backbone-config",
        type=str,
        default="src/anomalib/config/backbone_config.yaml",
        help="Path to a list of backbone configuration file",
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of random runs to sample from")
    parser.add_argument(
        "--k-shots",
        type=int,
        nargs="+",
    )
    parser.add_argument(
        "--augment",
        type=bool,
        default=False,
        help="Whether to augment the train set with augmentations defined on the config",
    )
    parser.add_argument(
        "--number-transforms",
        type=int,
        nargs="+",
        help="Number of transforms per k-shot",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        help="Eval/Test batch size per k-shot",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="List of input image size",
    )
    parser.add_argument(
        "--coreset-ratio",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="<DEBUG, INFO, WARNING, ERROR>",
    )
    args = parser.parse_args(raw_args)

    return args


def train(raw_args=None):
    """Train an anomaly classification or segmentation model based on a provided configuration file."""
    args = get_args(raw_args)
    configure_logger(level=args.log_level)

    if args.log_level == "ERROR":
        warnings.filterwarnings("ignore")

    # If one batch_size value is provided, apply this Eval/Test batch_size for all k_shots
    if len(args.batch_size) == 1:
        args.batch_size = args.batch_size * len(args.k_shots)
    assert len(args.k_shots) == len(args.batch_size)
    print(args.batch_size)

    # If one number_transforms value is provided, apply same number of transforms for all k_shots
    if args.augment:
        if len(args.number_transforms) == 1:
            args.number_transforms = args.number_transforms * len(args.k_shots)
    else:
        args.number_transforms = [None] * len(args.k_shots)

    assert len(args.k_shots) == len(args.number_transforms)
    print(args.number_transforms)

    transformer = Transformer(args.dataset)
    all_indexes = [idx for idx in range(len(transformer.transforms))]

    # create base config and seed
    config = get_configurable_parameters(
        model_name=args.model,
        dataset=args.dataset,
        config_path=args.config,
        backbone=args.backbone,
        backbone_config_path=args.backbone_config,
        k_shots=args.k_shots,
        runs=args.runs,
        augment=args.augment,
        number_transforms=args.number_transforms,
        batch_size=args.batch_size,
        image_size=args.image_size,
        coreset_ratio=args.coreset_ratio,
    )

    if config.project.get("seed") is not None:
        seed_everything(config.project.seed)

    augmentation_transforms = None
    if args.augment:
        augmentation_transforms = transformer.get_transforms_with_index(list_index=all_indexes)
        A.save(augmentation_transforms, Path(config.project.path) / "augmentations.json")

    if config.dataset.name == "mvtec":
        from anomalib.data.mvtec import MVTEC_CATEGORIES
        categories = MVTEC_CATEGORIES
    elif config.dataset.name == "visa":
        from anomalib.data.visa import VISA_CATEGORIES
        categories = VISA_CATEGORIES
    else:
        raise ValueError("Only mvtec and VisA are supported for now") 

    output_metrics = []

    # images have .jpg extension, but masks are pngs
    config.dataset.extensions = [".JPG", ".png"]

    # default_image_size = config.
    for k_shot, batch_size, nb_transforms in zip(
        config.project.k_shots, config.project.batch_size, config.project.number_transforms
    ):
        config.dataset.eval_batch_size = batch_size
        config.dataset.test_batch_size = batch_size
        for run in range(args.runs):
            for category in categories:
                print("\n\n")
                print(f"k-shot: {k_shot}")
                print(f"run: {run}")
                print(f"category: {category}")
                print("\n\n")
                train_step(
                    config=config,
                    number_transforms=nb_transforms,
                    augmentation_transforms=augmentation_transforms,
                    output_metrics=output_metrics,
                    k_shot=k_shot,
                    run=run,
                    category=category,
                )
                gc.collect()  # cleans unused memory, required to avoid OOM with GPU
                torch.cuda.empty_cache()

    plots_output_path = Path(config.project.path) / "plots/"
    plots_output_path.mkdir(parents=True, exist_ok=True)
    create_and_save_boxplots(pd.DataFrame(output_metrics), plots_output_path)

    return config.project.path


if __name__ == "__main__":
    train()
