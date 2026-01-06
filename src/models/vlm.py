import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model

class VisionLanguageModel(nn.Module):

    def __init__(self, backbone_id: str = None, device = None):
        super().__init__()
        self.device = device
        self.model_name = "VisionLanguageModel"
        self.backbone_id = backbone_id
        self.backbone = None
        self.processor = None

    def _inject_lora_layers(self, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1, target_modules: list = None):
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "up_proj", "down_proj"]

        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM
        )

        self.backbone = get_peft_model(self.backbone, lora_config)
        self.backbone.print_trainable_parameters()
        self.to(self.device)
