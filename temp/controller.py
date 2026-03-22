#------- This is the model for tracking quantization ----------
import torch
from torch.export import export, Dim
from samplenet import SampleCnn

#================================================================

model=SampleCnn().to('cpu')
model.eval()


B = Dim('batch',       min=1, max=128)
C = 3
_H = Dim('height',      min=1, max=129)
_W = Dim('width',       min=1, max=129)

H, W = _H*8, _W*8


sample= torch.randn(8,3,32,32)

ep = export( model,  args=(sample,),  dynamic_shapes={'x': {0: B, 1: C, 2: H, 3: W}})
ep.module()(sample)

#=========
print("\n"+ "-"*50 + "\n Printing Exported module \n"+ "-"*50)

#print(ep)
print(f"{'Tensor Name':<40} | {'Shape':<20}")
print("-" * 65)

for name, param in ep.module().named_parameters():
    print(f"{name:<40} | {str(list(param.shape)):<20}")

for name, buffer in ep.module().named_buffers():
    # Including buffers because BatchNorm running stats are often here
    print(f"{name:<40} | {str(list(buffer.shape)):<20} (buffer)")
