import torch
import torch.nn as nn

import numpy as np

import matplotlib.pyplot as plt

def plot_results(results, name='name'):
    plt.errorbar(results['flops'][::-1], results['mean_time'][::-1],  results['std_time'][::-1], marker='o', color='g', linestyle='dashdot')
    plt.xlabel("Log FLOPS")
    plt.ylabel("time")
    plt.legend(loc='upper left')
    plt.title(name)
    plt.xscale('log')
    plt.grid()
    plt.show()


def get_flops_and_memory(
    root_module, use_cached=False, method_name="_fetch_info", input_size=None , device="cpu"
):

    if not use_cached:
        assert (
            not input_size is None
        ), "Provide input_size to create a dummy tensor for FLOPS computation or use_cached=True"

        input_x = torch.randn(*input_size).to(device)
        root_module(input_x)

    mem = []
    fl = []

    def apply(module):
        for name, child in module.named_children():
            if hasattr(child, method_name):
                f, m = getattr(child, method_name)()
                fl.append(f)
                mem.append(m)
            else:
                apply(child)
        return fl, mem

    f, m = apply(root_module)
    return sum(f), sum(m)



class BasicModel(nn.Module):
    def __init__(
        self,
        BasicBlock, 
        features_in: int,
        features_out: int,
        layer_type: str = 'linear',
        kernel_size: int = 3,
        block_numbers: int = 2,
        dilation: int = 1,
        )-> None:
        super().__init__()

        layers = []
        for i in range(block_numbers):
            layers.append(BasicBlock(features_in, features_out, layer_type, kernel_size))

        self.model = nn.Sequential(*layers)

    
    def forward(self,x):
        return self.model(x)
    


def run_experiment(BasicBlock,
                layer_type, 
                features_in, 
                features_out, 
                dummy_tensor, 
                block_numbers, 
                device, 
                n_runs = 100):

    block = BasicModel(BasicBlock,
                    features_in=features_in, 
                    features_out=features_out, 
                    layer_type=layer_type,
                    block_numbers=block_numbers)
    block.eval()
    
    block.to(device)
    dummy_tensor = dummy_tensor.to(device)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings=np.zeros((n_runs,1))
    #GPU-WARM-UP
    for _ in range(10):
        _ = block(dummy_tensor)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(n_runs):
            starter.record()
            _ = block(dummy_tensor)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    
    flops, memory = get_flops_and_memory(block, use_cached=True)

    return {'mean_time':mean_syn, 'std_time':std_syn,  'flops':flops,'memory':memory}



def plot(results, name='name'):
    plt.errorbar(results['flops'][::-1], results['mean_time'][::-1],  results['std_time'][::-1], marker='o', color='g', linestyle='dashdot')
    plt.xlabel("Log FLOPS")
    plt.ylabel("time")
    plt.xscale('log')
    plt.grid()
    plt.show()

