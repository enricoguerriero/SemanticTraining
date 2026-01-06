def select_model(model_name: str, device: str = "cuda"):
    if model_name == "Qwen3VL":
        from src.models import Qwen3VL
        return Qwen3VL(device=device)
    else:
        raise ValueError(f"Model {model_name} not recognized.")