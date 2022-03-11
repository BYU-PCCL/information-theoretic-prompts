def get_device_map(gpus, n_layers):
    '''
    Given a list of gpus, make a dictionary map from GPU to layer, covering all layers evenly.
    params:
        gpus (list): list of gpus
        n_layers (int): number of layers
    return:
        device_map (dict): dictionary mapping from GPU to layer
    '''
    layers_per_gpu = n_layers // len(gpus)
    remainder = n_layers % len(gpus)
    cutoffs = [layers_per_gpu * i + min(i, remainder) for i in range(len(gpus))]
    cutoffs.append(n_layers)
    device_map = {
        gpu: [i for i in range(cutoffs[gpu_num], cutoffs[gpu_num + 1])] for gpu_num, gpu in enumerate(gpus)
    }
    return device_map

if __name__ == '__main__':
    test_gpus = [0, 1, 3, 4, 5, 6, 7]
    n_layers = 48
    device_map = get_device_map(test_gpus, n_layers)
    print(device_map)
