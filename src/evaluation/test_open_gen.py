"""
Evaluation script for LoRA fine-tuned models.
Computes loss, precision, recall, and F1 metrics over a test dataset.
Generates answers and stores them in a JSON file.
"""

from src.utils import select_model, lora_collate_fn, check_precision_in_text, check_recall_in_text
from src.data import EnrichedDataset

import logging
import json
from argparse import ArgumentParser
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from datetime import datetime


def evaluate_model(
    model,
    test_dataset,
    batch_size: int = 1,
    device: str = "cuda",
    logger=None,
    output_json_path: str = None
):
    """
    Evaluate a model on a test dataset.
    
    Args:
        model: The model to evaluate
        test_dataset: The test dataset
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        logger: Logger instance
        output_json_path: Path to save generated answers JSON file
    
    Returns:
        Dictionary containing evaluation metrics
    """
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lora_collate_fn
    )
    
    if logger:
        logger.info(f"Test DataLoader created with {len(test_loader)} batches")
    
    model.to(device)
    model.eval()
    
    total_loss = 0.0
    precision_totals = {}
    recall_totals = {}
    precision_counts = {}
    recall_counts = {}
    
    # Dictionary to store video_path -> generated_answer
    generated_answers = {}
    
    if logger:
        logger.info(" --- Starting Evaluation --- ")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values_videos = batch['pixel_values_videos'].to(device)
            labels = batch['labels'].to(device)
            video_grid_thw = batch['video_grid_thw'].to(device)
            input_len = batch['prompt_len'].to(device)
            
            # Compute loss
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                labels=labels
            )
            total_loss += outputs.loss.item()
            
            # Generate answers for metrics computation
            prompt_only_ids = []
            prompt_only_masks = []
            for i in range(len(input_ids)):
                split_index = input_len[i]
                full_sequence = input_ids[i]
                prompt_only = full_sequence[:split_index]
                prompt_only_ids.append(prompt_only)
                full_mask = attention_mask[i]
                prompt_only_mask = full_mask[:split_index]
                prompt_only_masks.append(prompt_only_mask)
            
            prompt_only_ids = pad_sequence(
                prompt_only_ids,
                batch_first=True,
                padding_value=151643  # PAD_TOKEN_ID for Qwen3VL
            ).to(device)
            prompt_only_masks = pad_sequence(
                prompt_only_masks,
                batch_first=True,
                padding_value=0
            ).to(device)
            
            generated_ids = model.generate(
                input_ids=prompt_only_ids,
                attention_mask=prompt_only_masks,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw
            )
            
            generated_text = model.processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Get video paths and ground truth keywords from the batch
            # We need to retrieve video paths from the dataset
            batch_start_idx = batch_idx * batch_size
            batch_video_paths = []
            for i in range(len(generated_text)):
                dataset_idx = batch_start_idx + i
                if dataset_idx < len(test_dataset):
                    video_path = test_dataset.videos[dataset_idx]
                    batch_video_paths.append(video_path)
            
            ground_truth_keywords_list = batch["ground_truth_keywords"]
            
            for idx, (gen_text, gt_keywords) in enumerate(zip(generated_text, ground_truth_keywords_list)):
                # Extract the assistant response
                gen_text_clean = gen_text.split("assistant\n")[-1].strip()
                
                if logger:
                    logger.debug(f"Generated text: {gen_text_clean}")
                    logger.debug(f"Ground truth keywords: {gt_keywords}")
                
                # Store in generated answers dictionary
                if idx < len(batch_video_paths):
                    video_path = batch_video_paths[idx]
                    generated_answers[video_path] = gen_text_clean
                
                # Compute precision and recall
                precision_dict = check_precision_in_text(gen_text_clean, gt_keywords)
                recall_dict = check_recall_in_text(gen_text_clean, gt_keywords)
                
                for key, value in precision_dict.items():
                    precision_totals[key] = precision_totals.get(key, 0) + value
                    precision_counts[key] = precision_counts.get(key, 0) + 1
                for key, value in recall_dict.items():
                    recall_totals[key] = recall_totals.get(key, 0) + value
                    recall_counts[key] = recall_counts.get(key, 0) + 1
    
    # Compute average metrics
    avg_loss = total_loss / len(test_loader)
    precision_avg = {k: v / precision_counts.get(k, 1) for k, v in precision_totals.items()}
    recall_avg = {k: v / recall_counts.get(k, 1) for k, v in recall_totals.items()}
    
    # Build final metrics dictionary
    final_metrics = {"Test Loss": avg_loss}
    
    all_classes = set(precision_totals.keys()).union(set(recall_totals.keys()))
    for cls in all_classes:
        p = precision_avg.get(cls, 0.0)
        r = recall_avg.get(cls, 0.0)
        f1 = (2 * p * r / (p + r + 1e-8)) if (p + r) > 0 else 0.0
        final_metrics[f"{cls}/Precision"] = p
        final_metrics[f"{cls}/Recall"] = r
        final_metrics[f"{cls}/F1-Score"] = f1
    
    # Compute macro averages
    if all_classes:
        macro_precision = sum(precision_avg.values()) / len(precision_avg) if precision_avg else 0.0
        macro_recall = sum(recall_avg.values()) / len(recall_avg) if recall_avg else 0.0
        macro_f1 = (2 * macro_precision * macro_recall / (macro_precision + macro_recall + 1e-8)) if (macro_precision + macro_recall) > 0 else 0.0
        final_metrics["Macro/Precision"] = macro_precision
        final_metrics["Macro/Recall"] = macro_recall
        final_metrics["Macro/F1-Score"] = macro_f1
    
    # Log results
    if logger:
        logger.info(f"Test Loss: {avg_loss:.4f}")
        logger.info(f"Precision per class: {precision_avg}")
        logger.info(f"Recall per class: {recall_avg}")
        if all_classes:
            logger.info(f"Macro Precision: {macro_precision:.4f}")
            logger.info(f"Macro Recall: {macro_recall:.4f}")
            logger.info(f"Macro F1-Score: {macro_f1:.4f}")
    
    # Save generated answers to JSON
    if output_json_path:
        with open(output_json_path, 'w') as f:
            json.dump(generated_answers, f, indent=2)
        if logger:
            logger.info(f"Generated answers saved to {output_json_path}")
    
    return final_metrics, generated_answers


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluation Script for LoRA Fine-tuned Models")
    parser.add_argument('--model', type=str, required=True, help='Base model name (e.g., Qwen3VL)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the LoRA fine-tuned model weights')
    parser.add_argument('--config', type=str, default="src/evaluation/eval_config.yaml", help='Path to evaluation config file')
    parser.add_argument('--output_json', type=str, default=None, help='Path to save generated answers JSON')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    model_name = args.model
    model_path = args.model_path
    config_path = args.config
    output_json = args.output_json
    debug_mode = args.debug
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if debug_mode else logging.INFO,
        format='[%(levelname)s]: %(message)s'
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting evaluation for model: {model_name}")
    logger.info(f"Loading weights from: {model_path}")
    
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.debug(f"Loaded configuration: {config}")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model info from checkpoint
    model_name = checkpoint['model_name']
    backbone_id = checkpoint['backbone_id']
    lora_config = checkpoint['lora_config']
    
    logger.info(f"Model: {model_name}")
    logger.info(f"Backbone: {backbone_id}")
    logger.debug(f"LoRA config: {lora_config}")
    
    # Load base model
    model = select_model(model_name, device=device)
    logger.info(f"Base model {model_name} loaded.")
    
    # Inject LoRA layers with config from checkpoint
    model._inject_lora_layers(
        r=lora_config['lora_r'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        target_modules=lora_config['lora_target_modules']
    )
    logger.debug("LoRA layers injected into the model.")
    
    # Load fine-tuned weights
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded fine-tuned weights from {model_path}")
    
    # Get prompt from config
    prompt = config.get("prompt", "Describe the content of the video.")
    logger.debug(f"Using prompt: {prompt}")
    
    # Load test dataset
    test_dataset = EnrichedDataset(
        video_csv=config.get("test_data", "data/test.csv"),
        prompt_template=prompt,
        processor=model.processor,
        num_frames=config.get("num_frames", None),
        enrichment_csv=config.get("enrichment_data", "src/data/enrichment.csv")
    )
    logger.info(f"Test dataset loaded with {len(test_dataset)} samples.")
    
    # Set default output path if not provided
    if output_json is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_json = f"evaluation_results_{model_name}_{timestamp}.json"
    
    # Run evaluation
    metrics, generated_answers = evaluate_model(
        model=model,
        test_dataset=test_dataset,
        batch_size=config.get("batch_size", 1),
        device=device,
        logger=logger,
        output_json_path=output_json
    )
    
    # Print final results
    logger.info(" --- Final Evaluation Results --- ")
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.4f}")
    
    # Save metrics to JSON as well
    metrics_json_path = output_json.replace('.json', '_metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_json_path}")
    
    logger.info("Evaluation completed.")
