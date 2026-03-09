import timm
import torch
import torch.nn as nn

ARCH_MAP = {
    # Vision Transformers
    "visiontransformer": "vit_base_patch16_224",
    "vit": "vit_base_patch16_224",

    # Swin Transformers
    "swin": "swin_base_patch4_window7_224",
    "swinv2": "swinv2_base_window8_256",
    "swin_t": "swin_tiny_patch4_window7_224",   # ✅ ADD THIS

    # ConvNeXt
    "convnext": "convnext_base",
    "convnext_v2": "convnextv2_base",

    # EfficientNet
    "efficientnet": "efficientnet_b0",
    "effnet_b0": "efficientnet_b0",
    "effnet_b3": "efficientnet_b3"
}




def build_backbone(name, num_classes):
    name = name.lower()

    if name not in ARCH_MAP:
        raise ValueError(f"Unsupported architecture: {name}")

    model = timm.create_model(
        ARCH_MAP[name],
        pretrained=True,
        num_classes=num_classes
    )
    return model


class WeightedSoftVotingEnsemble(nn.Module):
    def __init__(self, models, weights, num_classes):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = torch.tensor(weights).view(-1, 1)
        self.num_classes = num_classes

    def forward(self, x):
        probs = []
        for model in self.models:
            logits = model(x)
            probs.append(torch.softmax(logits, dim=1))

        probs = torch.stack(probs)
        weighted = probs * self.weights.to(probs.device)
        return weighted.sum(dim=0)