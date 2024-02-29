import time

import numpy as np
import pandas as pd
import torch
from loguru import logger


# from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete, Compose

from src.inference.metrics import get_sample_metrics_from_np_arrays
from src.utils.metrics_utils import (
    check_metric_validity,
    check_metric_types,
    get_metrics_to_eval,
)
from src.utils.train_utils import init_epoch_dict, get_timings_per_epoch


def evaluate_datasets_per_epoch(
    model,
    device,
    epoch,
    dataloaders,
    training_config,
    metric_dict,
    eval_config,
    split_name,
):
    epoch_metrics = init_epoch_dict(
        epoch, dataloaders[split_name], split_name=split_name
    )
    for j, dataset_name in enumerate(dataloaders[split_name].keys()):
        dataloader = dataloaders[split_name][dataset_name]
        epoch_metrics[dataset_name] = evaluate_1_epoch(
            dataloader,
            model,
            split_name,
            dataset_name,
            training_config,
            metric_dict,
            eval_config,
            device,
            epoch_metrics[dataset_name],
        )

    return epoch_metrics


def evaluate_1_epoch(
    dataloader,
    model,
    split_name,
    dataset_name,
    training_config,
    metric_dict,
    eval_config,
    device,
    epoch_metrics_per_dataset,
    return_probs=False,  # you can post-training do diverse ensembles with the probs
):
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # https://github.com/Project-MONAI/tutorials/blob/2183d45f48c53924b291a16d72f8f0e0b29179f2/acceleration/distributed_training/brats_training_ddp.py#L341
    model.eval()
    batch_evals = {}
    epoch_start = time.time()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if training_config["PRECISION"] == "AMP":
                with torch.cuda.amp.autocast():
                    val_outputs = sliding_window_inference(
                        inputs=batch_data["image"].to(device), **metric_dict
                    )
            else:
                val_outputs = sliding_window_inference(
                    inputs=batch_data["image"].to(device), **metric_dict
                )

            # y_pred_proba = [i for i in decollate_batch(val_outputs)]
            # y_pred = [post_trans(i) for i in decollate_batch(val_outputs)]
            y_pred = post_trans(val_outputs)

            batch_evals = evaluate_batch(
                y_pred=y_pred,
                y_pred_proba=val_outputs,
                y=batch_data["label"].to(device),
                metadata=batch_data["metadata"],
                device=device,
                batch_sz=batch_data["image"].shape[0],
                batch_evals=batch_evals,
                eval_config=eval_config,
            )

        epoch_metrics_per_dataset = get_epoch_metrics(
            epoch_metrics_per_dataset, batch_evals
        )

        epoch_metrics_per_dataset = get_timings_per_epoch(
            metadata_dict=epoch_metrics_per_dataset,
            epoch_start=epoch_start,
            no_batches=batch_idx + 1,
            mean_batch_sz=float(np.mean(batch_evals["batch_sz"])),
        )

    return epoch_metrics_per_dataset


def get_epoch_metrics(epoch_metrics_per_dataset, batch_evals):
    for metric_name in batch_evals["metric_names"]:
        values = batch_evals["resseg"][metric_name].values
        epoch_metrics_per_dataset["scalars"][metric_name] = np.mean(values)

    return epoch_metrics_per_dataset


def evaluate_batch(
    y_pred, y_pred_proba, y, metadata, device, batch_sz, batch_evals, eval_config
):
    metrics_to_eval = get_metrics_to_eval(eval_config)
    measures_overlap, measures_boundary = check_metric_types(
        metrics_to_eval=metrics_to_eval
    )

    PE, _ = get_sample_metrics_from_np_arrays(
        y_pred,
        y_pred_proba,
        y,
        metadata,
        eval_config,
        measures_overlap,
        measures_boundary,
    )

    if len(batch_evals) == 0:
        # first batch
        batch_evals["resseg"] = PE.resseg
        batch_evals["batch_sz"] = [batch_sz]
        batch_evals["metric_names"] = measures_overlap + measures_boundary
    else:
        batch_evals["resseg"] = pd.concat([batch_evals["resseg"], PE.resseg], axis=0)
        batch_evals["batch_sz"].append(batch_sz)

    return batch_evals
