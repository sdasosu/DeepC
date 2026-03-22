# experiments/run.py
import sys
import torch
import random
from pathlib import Path
from torchinfo import summary
from torch.export import export

# add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import segmentation_models_pytorch as smp


#---------------------------------------------------------------------
NUM_CLASSES = 1
DUMMY_INPUT = (1, 3, 576, 576)
#---------------------------------------------------------------------


#---------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
#---------------------------------------------------------------------


#================================== CREATE THE MODEL =========================================
def create_model(model_name: str, encoder_name: str, in_ch: int = 3, num_classes: int = NUM_CLASSES):

    m = model_name.lower()

    if m == "unet++":
        return smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_ch,
            classes=num_classes,
            activation=None,
        )

    if m == "pspnet":
        return smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_ch,
            classes=num_classes,
            activation=None,
        )

    if m == "fpn":
        return smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_ch,
            classes=num_classes,
            activation=None,
        )

    if m == "deeplabv3+":
        return smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_ch,
            classes=num_classes,
            activation=None,
        )

    raise ValueError(f"Unknown model: {model_name}")
#=============================================================================================


#---------------------------------------------------------------------
def print_model_info(model_name: str, encoder_name: str, device: torch.device):
    model = create_model(model_name=model_name, encoder_name=encoder_name).to(device)
    model.eval()

    x = torch.randn(*DUMMY_INPUT, device=device)

    print("\n" + "=" * 100)
    print(f"label=1 | model={model_name} | encoder={encoder_name}")
    print("=" * 100)

    summary(
        model,
        input_size=DUMMY_INPUT,
        depth=2,
        col_names=("input_size", "output_size", "num_params"),
        device=str(device),
    )

    if encoder_name in ["efficientnet-b4", "timm-efficientnet-b4"]:
        traced = torch.jit.trace(model, x)
        print(traced.graph)
        del traced
    else:
        ep = export(model, (x,), strict=False)
        print(ep.graph_module.graph)
        del ep

    del model, x
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
#---------------------------------------------------------------------


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = ["unet++", "pspnet", "fpn", "deeplabv3+"]
    encoders = ["resnet50", "resnet101", "densenet121", "densenet201", "timm-efficientnet-b4"]

    for model_name in models:
        for encoder_name in encoders:
            print_model_info(model_name=model_name, encoder_name=encoder_name, device=device)


if __name__ == "__main__":
    main()