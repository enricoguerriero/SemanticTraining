import torch
from src.utils.collate_fns import pad_sequence

def lora_collate_fn(batch):
    
    input_ids_list = [item['input_ids'] for item in batch]
    attention_mask_list = [item['attention_mask'] for item in batch]
    pixel_values_videos_list = [item['pixel_values_videos'] for item in batch]
    video_grid_thw_list = [item['video_grid_thw'] for item in batch if 'video_grid_thw' in item]
    prompt_len_list = [item['prompt_len'] for item in batch if 'prompt_len' in item]
    ground_truth_keywords_list = [item['ground_truth_keywords'] for item in batch if 'ground_truth_keywords' in item]

    PAD_TOKEN_ID = 151643 # For Qwen3VL, check processor.tokenizer.pad_token_id

    input_ids_padded = pad_sequence(
        input_ids_list, 
        batch_first=True, 
        padding_value=PAD_TOKEN_ID
    )
    attention_mask_padded = pad_sequence(
        attention_mask_list, 
        batch_first=True, 
        padding_value=0
    )

    labels = input_ids_padded.clone()
    labels[labels == PAD_TOKEN_ID] = -100

    for i, prompt_len in enumerate(prompt_len_list):
        labels[i, :prompt_len] = -100

    pixel_values_stacked = torch.cat(pixel_values_videos_list)
    video_grid_stacked = torch.cat(video_grid_thw_list) if video_grid_thw_list else None

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'pixel_values_videos': pixel_values_stacked,
        'video_grid_thw': video_grid_stacked,
        'labels': labels,
        'ground_truth_keywords': ground_truth_keywords_list if ground_truth_keywords_list else None
    }