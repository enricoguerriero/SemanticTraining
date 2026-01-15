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
    
    def generate(self, input_ids, attention_mask, pixel_values_videos, video_grid_thw):
        return self.backbone.generate(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      pixel_values_videos=pixel_values_videos,
                                      video_grid_thw=video_grid_thw, 
                                      top_p=0.8,
                                      top_k=20,
                                      temperature=0.7,
                                      repetition_penalty=1.0,
                                      max_new_tokens=150,
                                      do_sample=False
                                      ) #from HF qwen3vl