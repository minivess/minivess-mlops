import os

import numpy as np
from loguru import logger
import monai
from monai.handlers import MetricsReloadedBinaryHandler
from monai.data import decollate_batch


def check_metric_validity(
    metric_name: str,
) -> monai.handlers.metrics_reloaded_handler.MetricsReloadedBinaryHandler:
    # We are now using only the Metrics Reloaded package, so if the name is not found from possible
    # metrics, it is not valid. Add MONAI metrics later if you want to use them as well
    try:
        metric_handler = MetricsReloadedBinaryHandler(metric_name)
    except Exception as e:
        logger.error(
            "Metric {} is not found from the MetricsReloaded package".format(
                metric_name
            )
        )
        raise NotImplementedError(
            "Metric {} is not found from the MetricsReloaded package".format(
                metric_name
            )
        )

    return metric_handler


def check_metric_types(metrics_to_eval):
    try:
        measures_overlap = metrics_to_eval["Overlap"]
    except:
        logger.error(
            "No overlap metrics to evaluate, add to your config, keys are: {}".format(
                metrics_to_eval.keys()
            )
        )
        raise IOError("No overlap metrics to evaluate, add to your config")

    try:
        measures_boundary = metrics_to_eval["Boundary"]
    except:
        logger.error(
            "No boundary metrics to evaluate, add to your config, keys are: {}".format(
                metrics_to_eval.keys()
            )
        )
        raise IOError("No boundary metrics to evaluate, add to your config")

    return measures_overlap, measures_boundary


def init_metrics_reloaded_dict(y_pred, y_pred_proba, y, metadata):
    # https://github.com/Project-MONAI/tutorials/blob/1783005849df6129dc389ee3e537851bc44ab10d/modules/metrics_reloaded/unet_evaluation.py#L33

    def get_paths_and_fnames(metadata):
        paths = []
        names = []
        for i in metadata["filepath_json"]:
            paths.append(i)
            name = os.path.split(i)[1].split(".")[0]
            names.append(name)
        return paths, names

    def check_list_items(list_in):
        for list_item in list_in:
            assert len(list_item.shape) <= 3, (
                "Should be a 2D or 3D image, "
                "not 5D tensor, or 4D image with the channel dim"
            )

    def convert_tensor_to_list_of_tensors(y_pred, y_pred_proba, y):
        y_pred_proba = [
            i[0, :, :, :] for i in decollate_batch(y_pred_proba)
        ]  # drop the channel
        check_list_items(y_pred_proba)
        y_pred = [i[0, :, :, :] for i in decollate_batch(y_pred)]  # drop the channel
        check_list_items(y_pred)
        y = [i[0, :, :, :] for i in decollate_batch(y)]  # drop the channel
        check_list_items(y)

        return y_pred, y_pred_proba, y

    def convert_numpy_to_list_of_tensors(array_in):
        assert len(array_in.shape) == 4, "Should be a 4D tensor, not {}".format(
            len(array_in.shape)
        )
        tensors_out: list = []
        no_batches = array_in.shape[0]
        for b in range(no_batches):
            tensors_out.append(array_in[b, :, :, :])
        return tensors_out

    def convert_numpy_arrays_to_list_of_tensors(y_pred, y_pred_proba, y):
        y_pred = convert_numpy_to_list_of_tensors(array_in=y_pred)
        y_pred_proba = convert_numpy_to_list_of_tensors(array_in=y_pred_proba)
        y = convert_numpy_to_list_of_tensors(array_in=y)
        return y_pred, y_pred_proba, y

    paths, names = get_paths_and_fnames(metadata)
    if isinstance(y_pred, monai.data.MetaTensor):
        # during training, these are MetaTensors
        y_pred, y_pred_proba, y = convert_tensor_to_list_of_tensors(
            y_pred, y_pred_proba, y
        )
    else:
        # during ensembling, these are numpy arrays
        if isinstance(y_pred, np.ndarray):
            y_pred, y_pred_proba, y = convert_numpy_arrays_to_list_of_tensors(
                y_pred, y_pred_proba, y
            )
        else:
            logger.error("y_pred is not a numpy array, but a {}".format(type(y_pred)))
            raise TypeError(
                "y_pred is not a numpy array, but a {}".format(type(y_pred))
            )

    dict_file = {}
    dict_file["pred_loc"] = y_pred
    dict_file["ref_loc"] = y
    dict_file["pred_prob"] = y_pred_proba
    dict_file["ref_class"] = y
    dict_file["pred_class"] = y_pred
    dict_file["list_values"] = [
        1
    ]  # corresponds to "label" in PE.resseg? if we had image-level classes here
    dict_file["file"] = paths
    dict_file["names"] = names

    return dict_file


def get_metrics_to_eval(eval_config):
    metrics_to_eval = {}
    for i, metric_type in enumerate(eval_config["METRICS"]):
        metrics_to_eval[metric_type] = []
        for j, metric_name in enumerate(eval_config["METRICS"][metric_type]):
            _ = check_metric_validity(
                metric_name=metric_name
            )  # do only once somewhere before?
            metrics_to_eval[metric_type].append(metric_name)

    return metrics_to_eval
