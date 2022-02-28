############### MOBILE NET LAYERS ############################################################
############### FROM https://github.com/Bisonai/mobilenetv3-tensorflow/blob/master/layers.py

def _make_divisible(v, divisor, min_value=None):
    """https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor

    return new_v

def get_layer(layer_name, layer_dict, default_layer):
    if layer_name is None:
        return default_layer

    if layer_name in layer_dict.keys():
        return layer_dict.get(layer_name)
    else:
        raise NotImplementedError(f"Layer [{layer_name}] is not implemented")

def get_layer_dtype(activation:str):
    if activation == "sigmoid" or activation == "softmax":
        return "float32"
    else:
        return None
