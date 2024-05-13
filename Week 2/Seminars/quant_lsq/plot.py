import matplotlib.pyplot as plt

def plot_weight_distribution(model, Quantizer, bitwidth=32):
    num_weights = len([(p,n) for (p,n) in model.named_parameters() if n.numel()>1])-1
    print(num_weights)
    fig, axes = plt.subplots(1,num_weights, figsize=(15, 5))

    quantizer = Quantizer(bitwidth)
    qmin, qmax = quantizer.thd_neg,  quantizer.thd_pos
    ind = 0
    for _, (name, param) in enumerate(model.named_parameters()):
        if param.dim() >1 :
            quantizer.init_from(param)
            dequantized = quantizer(param)
            dequantized = dequantized.detach().cpu().numpy()

            axes[ind].hist(dequantized.flatten(),  density=True, color = 'blue', alpha = 0.5,
                    edgecolor='black' if bitwidth <= 4 else None, bins=2**bitwidth)

            axes[ind].set_xlabel(name)
            axes[ind].set_ylabel('density')
            ind+=1

    fig.suptitle(f'Histogram of Weights (bitwidth={bitwidth} bits)')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.show()