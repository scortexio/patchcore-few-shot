import re
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch

from anomalib.data.augmentation import Transformer
from anomalib.models.patchcore.torch_model import PatchcoreModel


def create_dataframe_from_seeds_file(base_seeds_dir, categories):
    """
    Create a pandas dataframe with the seeds for each category, k-shot
    and run.
    """
    data_list = []

    for category in categories:
        with open(base_seeds_dir + f"{category}/selected_samples_per_run.txt", "r") as f:
            data = f.readlines()

        for line in data:
            run = line[0]
            k_shot = line.split(":")[0][2:]
            indexes = re.sub("\n", "", line).split(" ")[1:]
            filenames = [ind for ind in indexes]
            data_list.append({"category": category, "k_shot": int(k_shot), "run": int(run), "filenames": filenames})

    seed_dataframe = pd.DataFrame(data_list)

    return seed_dataframe


def get_indexes_from_filenames(train_dataframe, sampling_filenames):
    indexes = []
    train_file_paths = train_dataframe.image_path.values
    train_filenames = np.array([Path(train_file).stem for train_file in train_file_paths])

    for filename in sampling_filenames:
        index = np.argwhere(train_filenames == filename)[0][0]
        indexes.append(index)

    return indexes


def filter_dataframe_experiment(
    dataframe: pd.DataFrame, k_shot: int, run: int, category: str, filter_column: str
) -> float:
    """Filter a dataframe whose rows contain a set of columns of k_shot, run and category
    used for filtering and a filter_column where we want to extract the value from.

    """
    return dataframe[(dataframe.k_shot == k_shot) & (dataframe.run == run) & (dataframe.category == category)][
        filter_column
    ].values[0]


def get_embedding_size(
    backbone: str = "wide_resnet50_2",
    input_image_size: Union[int, Tuple[int, int]] = 224,
    layers: List[str] = ["layer2", "layer3"],
) -> Dict:
    """Get the embedding size extracted from each layers and also the aligned&concatenated one.

    Args:
        backbone_config_path: Path to the list of backbone configurations
        input_image_size: if provided, use this as input size for the model
                          if not provided, use the default input size of the model

    Returns:
        a dict of layer name and its corresponding output size
    """
    if isinstance(input_image_size, int):
        input_image_size = (input_image_size, input_image_size)

    model = PatchcoreModel(
        input_size=input_image_size,
        layers=layers,
        backbone=backbone,
    )
    image = torch.rand(1, 3, input_image_size[0], input_image_size[1])

    _ = model(image)
    return model.layer_and_embedding_size


def get_augmentation_combinations_from_transformer(transformer: Transformer) -> List[List[int]]:
    """Given a transformer which contains a list of augmentations. Return the indexes of different
    augmentation combination, concretely:
    + Indexes of all augmentations
    + Indexes of all augmentations after removed 1 augmentation.

    E.g Given augmentations = [A, B, C, D]
    => augmentation_combination = [
        [0, 1, 2, 3],
        [   1, 2, 3],
        [0,    2, 3],
        [0, 1,  , 3]
        [0, 1, 2,  ],

    ]

    Args:
        transformer (Transformer): the transformer which contains list of augmentations
    """
    all_indexes = [idx for idx in range(len(transformer.transforms))]
    print(f"all indexes: {all_indexes}")
    combinations = [all_indexes]
    for idx in all_indexes:
        all_indexes_copy = all_indexes.copy()
        all_indexes_copy.remove(idx)
        combinations.append(all_indexes_copy)
    print(f"All combinations: {combinations}")

    return combinations
