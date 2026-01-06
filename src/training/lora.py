from src.utils import select_model, lora_collate_fn, check_precision_in_text, check_recall_in_text
from src.data import EnrichedDataset

import logging
from argparse import ArgumentParser
import yaml
import torch
import wandb
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence

def LoRA_training(
        model,
        train_dataset,
        val_dataset,
        batch_size: int = 1,
        num_epochs: int = 3,
        learning_rate: float = 1e-4,
        gradient_accumulation_steps: int = 16,
        device: str = "cuda",
        validation_interval: int = None,
        checkpoint_path: str = None,
        save_path: str = None,
        logger = None,
        wandb_run = None
    ):
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lora_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lora_collate_fn
    )
    if logger:
        logger.debug(f"DataLoaders created: train size {len(train_loader)}, val size {len(val_loader)}")

    model.to(device)
    model.backbone.gradient_checkpointing_enable()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    if logger:
        logger.debug(f"Total training steps: {total_steps}")
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    global_step = 0

    for epoch in range(num_epochs):
        if logger:
            logger.info(f" --- Starting epoch {epoch + 1}/{num_epochs} --- ")
        progress_bar = tqdm(train_loader, desc=f"Training epoch {epoch + 1}/{num_epochs}")
        current_loss = 0.0
        optimizer.zero_grad()

        LoRA_validate(model=model,
                      val_loader=val_loader,
                      device=device,
                      logger=logger,
                      wandb_run=wandb_run)

        for step, batch in enumerate(progress_bar):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values_videos = batch['pixel_values_videos'].to(device)
            labels = batch['labels'].to(device)
            video_grid_thw = batch['video_grid_thw'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                labels=labels
            )

            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            current_loss += loss.item()

            if (step + 1) % gradient_accumulation_steps == 0:
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                current_loss = 0.0
                global_step += 1
                progress_bar.set_postfix({"Global Step": global_step, "Loss": loss.item() * gradient_accumulation_steps})

                if wandb_run:
                    wandb_run.log({"Training Loss": loss.item() * gradient_accumulation_steps, "Global Step": global_step})

                if validation_interval and global_step % validation_interval == 0:
                    LoRA_validate(model=model,
                                  val_loader=val_loader,
                                  device=device,
                                  logger=logger,
                                  wandb_run=wandb_run)
                    if checkpoint_path:
                        ckpt_file = f"{checkpoint_path}/{model_name}_lora_checkpoint_step_{global_step}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                        torch.save(model.state_dict(), ckpt_file)
                        if logger:
                            logger.info(f"Saved checkpoint at {ckpt_file}")
        
        if checkpoint_path:
            epoch_file = f"{checkpoint_path}/{model.model_name}_lora_epoch_{epoch + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            torch.save(model.state_dict(), epoch_file)
            if logger:
                logger.info(f"Saved model after epoch {epoch + 1} at {epoch_file}")

        LoRA_validate(model=model,
                      val_loader=val_loader,
                      device=device,
                      logger=logger,
                      wandb_run=wandb_run)

    if save_path:
        final_file = f"{save_path}/{model.model_name}_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        torch.save(model.state_dict(), final_file)
        if logger:
            logger.info(f"Saved final model at {final_file}")


def LoRA_validate(model,
                  val_loader,
                  device: str = "cuda",
                  logger = None,
                  wandb_run = None):
    model.eval()
    total_val_loss = 0.0
    precision_totals = {}
    recall_totals = {}
    precision_counts = {}
    recall_counts = {}

    logger.info(" --- Starting validation --- ")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values_videos = batch['pixel_values_videos'].to(device)
            labels = batch['labels'].to(device)
            video_grid_thw = batch['video_grid_thw'].to(device)
            input_len = batch['prompt_len'].to(device)

            val_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                labels=labels
            )
            total_val_loss += val_outputs.loss.item()

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
                video_grid_thw=video_grid_thw,
                max_new_tokens=50
            )

            generated_text = model.processor.batch_decode(generated_ids, skip_special_tokens=True).split("assistant\n")[-1].strip()

            ground_truth_keywords_list = batch["ground_truth_keywords"]
            for gen_text, gt_keywords in zip(generated_text, ground_truth_keywords_list):

                logger.debug(f"Generated text: {gen_text}")
                logger.debug(f"Ground truth keywords: {gt_keywords}")
                
                precision_dict = check_precision_in_text(gen_text, gt_keywords)
                recall_dict = check_recall_in_text(gen_text, gt_keywords)

                for key, value in precision_dict.items():
                    precision_totals[key] = precision_totals.get(key, 0) + value
                    precision_counts[key] = precision_counts.get(key, 0) + 1
                for key, value in recall_dict.items():
                    recall_totals[key] = recall_totals.get(key, 0) + value
                    recall_counts[key] = recall_counts.get(key, 0) + 1

    avg_val_loss = total_val_loss / len(val_loader)
    precision_avg = {k: v / precision_counts.get(k, 1) for k, v in precision_totals.items()}
    recall_avg = {k: v / recall_counts.get(k, 1) for k, v in recall_totals.items()}
    logger.info(f"Validation Loss: {avg_val_loss}")
    logger.debug(f"Precision per class: {precision_avg}")
    logger.debug(f"Recall per class: {recall_avg}")
    final_metrics = {}
    all_classes = set(precision_totals.keys()).union(set(recall_totals.keys()))
    for cls in all_classes:
        final_metrics.update({
            f"{cls}/Precision": precision_avg.get(cls, 0.0),
            f"{cls}/Recall": recall_avg.get(cls, 0.0),
            f"{cls}/F1-Score": (2 * precision_avg.get(cls, 0.0) * recall_avg.get(cls, 0.0) / (precision_avg.get(cls, 0.0) + recall_avg.get(cls, 0.0) + 1e-8)) if (precision_avg.get(cls, 0.0) + recall_avg.get(cls, 0.0)) > 0 else 0.0
        })
    final_metrics["Validation Loss"] = avg_val_loss
    if wandb_run:
        wandb_run.log(final_metrics)
            

if __name__ == "__main__":

    parser = ArgumentParser(description="LoRA Training Script")
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()
    model_name = args.model
    debug_mode = args.debug

    logging.basicConfig(level=logging.DEBUG if debug_mode else logging.INFO, format='[%(levelname)s]: %(message)s')
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)

    logger.info(f"Starting LoRA training for model: {model_name}")

    with open("src/training/lora_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    logger.debug(f"Loaded configuration: {config}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"Using device: {device}")
    model = select_model(model_name, device=device)
    logger.debug(f"Model {model_name} selected and loaded.")
    logger.debug(f"Model architecture: {model}")

    wandb_run = wandb.init(
        project="Semantic-training",
        name=f"LoRA_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=config,
        save_code=True
    )

    prompt = config.get("prompt", "Describe the content of the video.")
    logger.debug(f"Using prompt: {prompt}")

    train_dataset = EnrichedDataset(
        video_csv=config.get("train_data", "train.csv"),
        prompt_template=prompt,
        processor=model.processor,
        num_frames=config.get("num_frames", None),
        enrichment_csv=config.get("enrichment_data", "enriched_classes.csv")
    )
    logger.info(f"Training dataset loaded with {len(train_dataset)} samples.")
    val_dataset = EnrichedDataset(
        video_csv=config.get("val_data", "val.csv"),
        prompt_template=prompt,
        processor=model.processor,
        num_frames=config.get("num_frames", None),
        enrichment_csv=config.get("enrichment_data", "enriched_classes.csv")
    )
    logger.info(f"Validation dataset loaded with {len(val_dataset)} samples.")

    model._inject_lora_layers(
        r=config.get("lora_r", 8),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0.1),
        target_modules=config.get("lora_target_modules", None)
    )
    logger.debug(f"LoRA layers injected into the model: {config.get('lora_target_modules', None)}")
    
    LoRA_training(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config.get("batch_size", 1),
        num_epochs=config.get("num_epochs", 3),
        learning_rate=config.get("learning_rate", 1e-4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 16),
        device=device,
        validation_interval=config.get("validation_interval", None),
        checkpoint_path=config.get("checkpoint_path", None),
        save_path=config.get("save_path", None),
        logger=logger,
        wandb_run=wandb_run
    )

    logger.info("LoRA training completed.")

