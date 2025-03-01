import torch
import torch.nn as nn
import os
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple, Set
from peft import load_peft_weights, set_peft_model_state_dict

parser = argparse.ArgumentParser(description="The main config of the adapter merging.")
parser.add_argument(
    "--candidate_lora", type=str, required=True, help="The path to the PEFT adapter"
)
parser.add_argument("--merge", type=str, required=True, help="The config of merging")
args = parser.parse_args()
with open(args.candidate_lora, "r") as f:
    file_config = json.load(f)
with open(args.merge, "r") as f:
    merge_config = json.load(f)


def dare_t(paramters: torch.Tensor, p: float):
    """
    Drop And REscale.
    DARE is a method before merge the parameters by dropout some parameters.
    Args:
        paramters (torch.Tensor): The tensor to filter.
        p (float): The percentage of parameters that drop.
    """
    num_zero_elements = int(p * paramters.numel())
    zero_indices = torch.randperm(paramters.numel())[:num_zero_elements]
    tensor_filtered = paramters.clone().view(-1)
    tensor_filtered[zero_indices] = 0.00
    tensor_filtered = tensor_filtered.view(paramters.shape)
    tensor_filtered = tensor_filtered / (1 - p)
    return tensor_filtered


def filter_values_max(tensor: torch.Tensor, p: float = 0.4) -> torch.Tensor:
    """
    Filters the values of a tensor by a percentage threshold. The threshold is calculated as a percentile of the value.

    Args:
        tensor (torch.Tensor): The tensor to filter.
        p (float): The percentage of parameters that remain.

    Returns:
        torch.Tensor: The filtered tensor.
    Example:
    torch.tensor([[ 1,  3, 5],            tensor([[ 0, 0, 5.],
                  [-4,  5, 6],      --->          [-4, 5, 6.],
                  [ 7, -8, 1]])                   [ 7,-8, 0.]]
    """
    max_values, _ = tensor.abs().max(dim=0)
    diff = max_values - tensor.abs()
    threshold = p * max_values
    mask = diff <= threshold
    filtered_tensor = tensor * mask.float()
    return filtered_tensor


def filter_values_percent(tensor: torch.Tensor, p: float = 0.4) -> torch.Tensor:
    """
    Filters the values of a tensor by a percentage threshold. The threshold is calculated as a percentage of the column.
    Args:
        tensor (torch.Tensor): The tensor to filter.
        p (float): The percentage of parameters that remain.
    Returns:
        torch.Tensor: The filtered tensor.
    """
    sorted_values, _ = torch.sort(tensor.abs(), dim=0)
    threshold_index = int(p * sorted_values.size(0))
    threshold = sorted_values[threshold_index]
    mask = tensor.abs() >= threshold
    filtered_tensor = tensor * mask.float()
    return filtered_tensor


def filter_sign(tensor: torch.Tensor) -> torch.Tensor:
    """
    Filters the values of a tensor by sign.

    Args:
        tensor (torch.Tensor): The tensor to filter.

    Returns:
        torch.Tensor: The filtered tensor.

    Example:
    torch.tensor([[ 0,  0, 5],            tensor([[ 0, 0, 5.],
                  [-4,  5, 6],      --->          [ 0, 0, 6.],
                  [ 7, -8, 0]])                   [ 7,-8, 0.]]
    """
    sum_values = tensor.sum(dim=0)
    direction = sum_values.sign()
    mask = (tensor.sign() == direction).float()
    filtered_tensor = tensor * mask
    return filtered_tensor, direction


def ties_merging(
    candidates: torch.Tensor, top_p: float = 0.3, merge_func: str = "mean"
):
    """
    Merges the adapters at the same target module of a flat tensor by ties.
    Args:
        candidates (torch.Tensor): A 2D tensor of shape (n, d) where n is the number of candidate and d is the number of parameters per candidate.
        top_k (float, optional): Should be a float between 0 and 1. the top-K% parameters to be keeped. if 1, keep all the parameters).
        merge_func (str, optional): The merge function to use for aggregating the parameters.Can be "mean", "sum", or "max". Defaults to "mean".

    Returns:
        torch.Tensor: A 1D tensor of shape (d,) stand for the merged parameter.

    """
    # step1 filter the top_k% parameters
    candidates = filter_values_percent(candidates, top_p)
    # step2 select sign of the parameters
    candidates, sign = filter_sign(candidates)
    # step3 merge the parameters
    if merge_func == "mean":
        non_zero_counts = (candidates != 0).sum(dim=0).float()
        return candidates.sum(dim=0) / non_zero_counts
    elif merge_func == "sum":
        return candidates.sum(dim=0)
    elif merge_func == "max":
        return candidates.abs().max(dim=0).values * sign


def average_merging(
    candidates: torch.Tensor,
):
    pass


def parameter_edit(
    candidate: torch.Tensor, method: str, dare: bool = False, dare_rate: float = 0.0005
):
    """
    Merges the parameters at the same target of a union flat tensor.

    Args:
        candidate (torch.Tensor): A 2D tensor of candidate adapters' value shaped by (n, d).
        method (str): The method used to merge the adapters. The options are "ties" and ....
        dare (bool): Whether to use the DARE method to merge the adapters.

    Returns:
        torch.Tensor: A 1D tensor of flat adapter shaped by (d,).
    """
    if dare:
        candidate = dare_t(candidate, p=dare_rate)
    if method == "ties":
        return ties_merging(candidate, top_p=0.4, merge_func="mean")
    if method == "average":
        return average_merging(candidate)


def Adapter_merge(adapters: List[Dict[str, torch.Tensor]], method: str, dare: float):
    """
    Merges the adapters at the same place of a flat tensor. This method is a paramter-level ensemble method.

    Args:
        adapters (List[Dict[str, torch.Tensor]]): A list of dictionaries where each dictionary contains the adapters of a model.
        method (str): The method used to merge the adapters. The options are "ties" and ....
        dare (bool): Whether to use the DARE method to merge the adapters. !!!This hyperparameter is very sensitive!!!

    Returns:
        Dict[str, torch.Tensor]: A dictionary where the key is the module name and the value is the merged adapter.
    """

    def reshape_(*values):
        result = torch.stack([value.view(1, -1) for value in values], dim=0)
        return result

    merged_adapters = {}
    for module_name in adapters[0].keys():
        shape = adapters[0][module_name].shape
        values = [adapter[module_name] for adapter in adapters]
        merged_adapters[module_name] = reshape_(*values)
        merged_adapters[module_name] = merged_adapters[module_name].squeeze(1)
        merged_adapters[module_name] = parameter_edit(
            merged_adapters[module_name], method, False, dare
        )
        merged_adapters[module_name] = merged_adapters[module_name].view(shape)
    return merged_adapters


if __name__ == "__main__":
    target_lora_path = file_config["target_adapter"]
    related_lora_path_list = file_config["related_adapter"]
    target_lora = load_peft_weights(target_lora_path)
    related_lora = [load_peft_weights(path) for path in related_lora_path_list]

    target_lora = dict(target_lora)
    related_lora = [dict(lora) for lora in related_lora]

    related_lora.append(target_lora)
    from utils import scaleAdapter

    for i in range(len(related_lora)):
        related_lora[i] = scaleAdapter(related_lora[i], merge_config["weights"][i])
    print("merging adapter with ", target_lora_path, "and", related_lora_path_list)
    merged_Adapter = Adapter_merge(
        related_lora, method=merge_config["method"], dare=merge_config["is_dare"]
    )
    os.makedirs(file_config["save_path"], exist_ok=True)
    torch.save(merged_Adapter, file_config["save_path"] + os.sep + "adapter_model.bin")
