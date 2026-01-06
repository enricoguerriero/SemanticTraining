from .vlm import VisionLanguageModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

class Qwen3VL(VisionLanguageModel):

    def __init__(self, backbone_id: str = "Qwen/Qwen3-VL-8B-Instruct", device: str = "cuda"):
        super().__init__(
                    backbone_id=backbone_id, 
                    device=device
                    )
        self.model_name = "Qwen3VL"
        self.device = device
        self.backbone = Qwen3VLForConditionalGeneration.from_pretrained(backbone_id)
        self.processor = AutoProcessor.from_pretrained(backbone_id)
    
