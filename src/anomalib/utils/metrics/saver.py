from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.pyplot import close
import numpy as np
import pandas as pd
import seaborn as sns

from anomalib.utils.general import filter_dataframe_experiment


def create_and_save_boxplots(results_dataframe: pd.DataFrame, output_path: Path):
    """Create and save box plots from the outputs of a training / testing experiment.

    Args:
        results_dataframe (pd.DataFrame): Pandas dataframe with f1-max results for each run,
            k-shot, category and task.
        output_path (Path): path where to save the output images
    """
    # get k-shot, runs and categories
    k_shot_options = sorted(results_dataframe.k_shot.unique())
    runs = sorted(results_dataframe.run.unique())
    categories = sorted(results_dataframe.category.unique())

    for category in categories:
        metric_for_category = np.zeros((len(k_shot_options), len(runs), 2))
        for k_idx, k_shot in enumerate(k_shot_options):
            for run in runs:
                f1_classification = filter_dataframe_experiment(
                    results_dataframe, k_shot=k_shot, run=run, category=category, filter_column="image_F1Score"
                )

                f1_segmentation = filter_dataframe_experiment(
                    results_dataframe, k_shot=k_shot, run=run, category=category, filter_column="pixel_F1Score"
                )

                metric_for_category[k_idx, run] = [f1_classification, f1_segmentation]

        # create plot
        fig = plt.figure(figsize=(10, 6))
        plt.suptitle(f"{category}")
        plt.subplot(121)
        plt.title("Classification")
        median_data_class = [[np.median(m)] for m in metric_for_category[:, :, 0]]
        sns.pointplot(data=median_data_class, color="black")
        sns.boxplot(x=np.repeat(k_shot_options, len(runs)), y=metric_for_category[:, :, 0].flatten())
        plt.ylabel("F1-max")
        plt.xlabel("k-shot")
        plt.subplot(122)
        plt.title("Segmentation")
        median_data_seg = [[np.median(m)] for m in metric_for_category[:, :, 1]]
        sns.pointplot(data=median_data_seg, color="black")
        sns.boxplot(x=np.repeat(k_shot_options, len(runs)), y=metric_for_category[:, :, 1].flatten())
        plt.ylabel("F1-max")
        plt.xlabel("k-shot")

        # save plot
        fig.savefig(output_path / f"metric_variation_{category}.png")
        close("all")
