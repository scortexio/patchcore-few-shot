import pandas as pd
from scipy.integrate import simps

from anomalib.utils.general import filter_dataframe_experiment


def compute_competition_metric(results_dataframe: pd.DataFrame, metric_type: str = "average") -> float:
    """Compute the metric for the competition based on aggregating f1-max scores
    across tasks (classification, segmentations), runs, categories and finally computing
    the area under the f1-max vs k-shot curve.

    Args:
        results_dataframe (pd.DataFrame): Pandas dataframe with f1-max results for each run,
            k-shot, category and task.
        metric_type (str, optional): whether to compute average f1-max, f1-max-classification or
            f1-max-segmentations. Can choose between average, class or seg. Defaults to "average".

    Raises:
        ValueError: raised if not valid metric_type

    Returns:
        float: final metric computed
    """
    if metric_type not in ("average", "class", "seg"):
        raise ValueError("Not a valid metric_type parameter")

    # get k-shot, runs and categories
    k_shot_options = sorted(results_dataframe.k_shot.unique())
    runs = sorted(results_dataframe.run.unique())
    categories = sorted(results_dataframe.category.unique())

    metric_for_each_k_shot = []

    for k_shot in k_shot_options:
        # create mean metric over each category and runs
        mean_metric_over_categories = 0.0

        for category in categories:
            # initialize metric for the runs of each category
            run_mean_metric = 0.0
            for run in runs:
                f1_classification = filter_dataframe_experiment(
                    dataframe=results_dataframe,
                    k_shot=k_shot,
                    run=run,
                    category=category,
                    filter_column="image_F1Score",
                )

                f1_segmentation = filter_dataframe_experiment(
                    dataframe=results_dataframe,
                    k_shot=k_shot,
                    run=run,
                    category=category,
                    filter_column="pixel_F1Score",
                )

                if metric_type == "average":
                    # compute harmonic mean over classification and segmentation
                    run_mean_metric += 2 / ((1 / f1_classification) + (1 / f1_segmentation))
                elif metric_type == "class":
                    run_mean_metric += f1_classification
                elif metric_type == "seg":
                    run_mean_metric += f1_segmentation

            # average over runs
            mean_metric_over_categories += run_mean_metric / len(runs)

        # divide by total number of categories
        mean_metric_over_categories /= len(categories)

        metric_for_each_k_shot.append(mean_metric_over_categories)

    # integrate and normalize
    final_metric = simps(y=metric_for_each_k_shot, x=k_shot_options) / (k_shot_options[-1] - k_shot_options[0])

    return final_metric
