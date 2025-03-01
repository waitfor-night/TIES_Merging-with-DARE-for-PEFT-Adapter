import os
import torch
from typing import List
from peft import load_peft_weights

    

def Adapter_reader(adapter_url:str, adapter_name:List[str]):
    Adapter_Dict = {}
    for name in adapter_name:
        Adapter_Dict[name] = load_peft_weights(os.path.join(adapter_url,name))
    return Adapter_Dict


def flattenTaskVector(Adapter_name:dict[str,torch.tensor])->torch.tensor:
    TaskVector = torch.cat([Adapter_name[key].view(-1) for key in Adapter_name.keys()])
    return TaskVector

def scaleAdapter(Adapter:dict[str,torch.tensor], scale:float)->dict[str,torch.tensor]:
    """
    Args:
        Adapter (dict[str,torch.tensor]): A dictionary where the key is the module name and the value is the adapter.
        scale (float): The scale factor.
    """
    for key in Adapter.keys():
        Adapter[key] = Adapter[key]*scale
    return Adapter

def drawTaskVector_diff(Task_vector:List[torch.tensor], base_vector:torch.tensor)->torch.tensor:
    pass