import torch
import torch.nn as nn

#============= Helper Functions =============
def filter_outliers(T, C=1.4826, min_val=-2.0, max_val=2.0):
    _temp = T.clone().detach().float()
    flat = _temp.view(-1)
    median = torch.median(flat)
    mad = torch.median(torch.abs(flat - median)) * C
    z_score = (_temp - median) / (mad + 1e-9)
    _temp[torch.abs(z_score) > 4.0] = torch.clamp(_temp[torch.abs(z_score) > 4.0], min_val, max_val)
    return _temp

def quantize_tensor(T_clean, bit, qtype):
    if qtype.lower() in ['s', 'sym', 'symmetric']:
        max_range = float(2**(bit-1) - 1)
        scale = torch.max(torch.abs(T_clean)) / (max_range + 1e-9)
        t_quant = torch.clamp(torch.round(T_clean / (scale + 1e-9)), -max_range, max_range)
        return t_quant * scale
    
    elif qtype.lower() in ['a', 'asym', 'asymmetric']:
        q_min, q_max = 0.0, float(2**bit - 1)
        _min, _max = torch.min(T_clean), torch.max(T_clean)
        scale = (_max - _min) / (q_max - q_min + 1e-9)
        zp = torch.clamp(torch.round(q_min - (_min / (scale + 1e-9))), q_min, q_max)
        t_quant = torch.clamp(torch.round((T_clean / (scale + 1e-9)) + zp), q_min, q_max)
        return scale * (t_quant - zp)
    return T_clean

def base_quant(layer_node, bit, qtype):
    weights = layer_node.weight.data
    recovered = torch.zeros_like(weights)
    for i in range(weights.size(0)):
        clean = filter_outliers(weights[i])
        recovered[i] = quantize_tensor(clean, bit, qtype)
    return recovered

#============= Quantize Layers =============
def QuantCNN(layer, bit, qtype='s'): return base_quant(layer, bit, qtype)
def QuantLinear(layer, bit, qtype='s'): return base_quant(layer, bit, qtype)
def QuantEmbedding(layer, bit, qtype='s'): return base_quant(layer, bit, qtype)
def QuantConvTranspose(layer, bit, qtype='s'): return base_quant(layer, bit, qtype)
def QuantDepthwiseConv(layer, bit, qtype='s'): return base_quant(layer, bit, qtype)
def QuantPointwiseConv(layer, bit, qtype='s'): return base_quant(layer, bit, qtype)

#============= Main / Test Cases =============
if __name__ == "__main__":
    def report(name, original, recovered):
        mae = torch.mean(torch.abs(original - recovered))
        print(f"{name:25} | MAE: {mae.item():.6f} | Shape: {list(recovered.shape)}")

    # 1. Conv2d
    l_cnn = nn.Conv2d(3, 16, 3)
    l_cnn.weight.data[0] *= 10 # outlier
    report("Conv2d", l_cnn.weight.data, QuantCNN(l_cnn, 4, 's'))

    # 2. Linear
    l_lin = nn.Linear(128, 64)
    report("Linear", l_lin.weight.data, QuantLinear(l_lin, 8, 'a'))

    # 3. Embedding
    l_emb = nn.Embedding(100, 32)
    report("Embedding", l_emb.weight.data, QuantEmbedding(l_emb, 4, 's'))

    # 4. ConvTranspose2d
    l_tr = nn.ConvTranspose2d(16, 8, 3)
    report("ConvTranspose", l_tr.weight.data, QuantConvTranspose(l_tr, 4, 'a'))

    # 5. Depthwise Conv (groups == in_channels)
    l_dw = nn.Conv2d(16, 16, 3, groups=16)
    report("DepthwiseConv", l_dw.weight.data, QuantDepthwiseConv(l_dw, 4, 's'))

    # 6. Pointwise Conv (kernel size 1)
    l_pw = nn.Conv2d(16, 32, 1)
    report("PointwiseConv", l_pw.weight.data, QuantPointwiseConv(l_pw, 4, 'a'))
