"""
InternImage-Large wrapper for classification, compatible with the Ark+ Finetuning pipeline.

Uses HuggingFace transformers to load the InternImage backbone (with trust_remote_code).
DCNv3 CUDA kernels are optional — the model falls back to a pure-PyTorch implementation
automatically if they are not installed.

Supported HuggingFace model identifiers for InternImage-Large:
  - "OpenGVLab/internimage_l_22kto1k_384"  (IN-22K → IN-1K finetuned, 384×384)
  - "OpenGVLab/internimage_l_22k_384"      (IN-22K pretrained, 384×384)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class InternImageLarge(nn.Module):
    """
    InternImage-Large backbone + a linear classification head.

    The backbone is loaded via HuggingFace ``AutoModel`` (num_classes=0, no
    built-in head).  A global average pool over the last stage feature map
    followed by a ``nn.Linear`` produces the logits.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    pretrained : bool
        If True, load ImageNet-22K-to-1K pretrained weights from HuggingFace.
        If False, initialise the backbone with random weights.
    hf_model_name : str
        HuggingFace model identifier.  Defaults to the IN-22K→1K checkpoint.
    """

    # Last-stage channel count for InternImage-L (C1=160 → C4 = 160*8 = 1280)
    NUM_FEATURES = 1280

    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        hf_model_name: str = "OpenGVLab/internimage_l_22kto1k_384",
    ):
        super().__init__()

        if pretrained:
            # Load backbone with pretrained weights (classification head stripped)
            self.backbone = AutoModel.from_pretrained(
                hf_model_name, trust_remote_code=True
            )
        else:
            # Random init — same architecture, no pretrained weights
            config = AutoConfig.from_pretrained(
                hf_model_name, trust_remote_code=True
            )
            self.backbone = AutoModel.from_config(config, trust_remote_code=True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.NUM_FEATURES, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (B, 3, H, W)

        Returns
        -------
        logits : Tensor of shape (B, num_classes)
        """
        out = self.backbone(x)  # dict with 'hidden_states' list
        feat = out["hidden_states"][-1]  # (B, 1280, H/32, W/32)
        feat = self.avgpool(feat).flatten(1)  # (B, 1280)
        return self.head(feat)
