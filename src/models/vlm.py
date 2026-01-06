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

    def forward(self, input_ids, attention_mask, pixel_values_videos, video_grid_thw=None, labels=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            labels=labels
        )
        return outputs
    
    def generate(self, *args, **kwargs):
        return self.backbone.generate(**args, **kwargs)