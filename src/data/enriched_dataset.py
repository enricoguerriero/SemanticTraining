from .clip_dataset import ClipDataset
import pandas as pd
import re

class EnrichedDataset(ClipDataset):

    def __init__(self, video_csv: str, prompt_template: str, processor, num_frames: int, enrichment_csv: str):

        self.processor = processor
        self.prompt = prompt_template
        self.data = pd.read_csv(video_csv)
        self.num_frames = num_frames
        self.enrichment_data = pd.read_csv(enrichment_csv, skipinitialspace=True)
        self.target_classes = ["No Newborn", "Baby visible", "CPAP", "PPV", "Suction", "Stimulation extremities", "Stimulation backnates", "Stimulation trunk"]
        self.enriched_descriptions, self.binary_labels_map = self._build_labels()
        self.ground_truth_labels = self._from_labels_to_keywords(self.binary_labels_map)
        self.videos, self.indices = self._prepare_videos(self.data['video_path'].tolist(), self.num_frames)
        if self.num_frames is None:
            self.num_frames = max([len(idxs) for idxs in self.indices])
        
    def _from_labels_to_keywords(self, binary_labels_map):
        keywords_map = {}
        for video_path, label_series in binary_labels_map.items():
            active_keywords = label_series[label_series == 1].index.tolist()
            keywords_map[video_path] = active_keywords
        return keywords_map

    @staticmethod
    def _extract_labels(target_classes, video_path):

        results = {}
        for c in target_classes:
            results[c] = 0
            results[f"P-{c}"] = 0
        
        for c in target_classes:
            if re.search(f"P-{re.escape(c)}", video_path):
                results[f"P-{c}"] = 1
            if re.search(f"(?<!P-){re.escape(c)}", video_path):
                results[c] = 1
        
        return pd.Series(results)
    
    @staticmethod
    def _map_labels_to_descriptions(enrichment_data, labels):
        
        interventions = ["CPAP", "PPV", "Suction", "Stimulation extremities", "Stimulation backnates", "Stimulation trunk"]

        if labels["No Newborn"] == 1:
            description = enrichment_data.loc[
                enrichment_data["classes enriched"] == "No Newborn", "descriptions"
            ].iloc[0]
        elif labels["Baby visible"] == 1 and not any(labels[k] == 1 for k in (interventions + [f"P-{i}" for i in interventions])):
            description = enrichment_data.loc[
                enrichment_data["classes enriched"] == "Baby visible", "descriptions"
            ].iloc[0]
        else:
            description = enrichment_data.loc[
                enrichment_data["classes enriched"] == "Baby visible", "descriptions"
            ].iloc[0].split(";")[0] + "."
            for intervention in interventions:
                if labels[intervention] == 1:
                    desc_part = enrichment_data.loc[
                        enrichment_data["classes enriched"] == intervention, "descriptions"
                    ].iloc[0]
                    description += " " + desc_part
                if labels[f"P-{intervention}"] == 1:
                    desc_part = enrichment_data.loc[
                        enrichment_data["classes enriched"] == intervention, "descriptions"
                    ].iloc[0]
                    description += " In part of the clip: " + desc_part

        return description
    
    def _enrich_labels(self, video_path):
        labels = self._extract_labels(self.target_classes, video_path)
        self.raw_labels[video_path] = labels
        description = self._map_labels_to_descriptions(self.enrichment_data, labels)
        return description, labels
        

    def _build_labels(self):
        enriched_descriptions = {}
        labels = {}
        self.raw_labels = {}
        for idx, row in self.data.iterrows():
            video_path = row['video_path']
            enriched_description, label = self._enrich_labels(video_path)
            enriched_descriptions[video_path] = enriched_description
            labels[video_path] = label
        return enriched_descriptions, labels

    
    def __getitem__(self, idx):
        video_path = self.videos[idx]
        frame_indices = self.indices[idx]
        frames, fps, total_frames = self._read_frames_at_indices(video_path, frame_indices)
        enriched_description = self.enriched_descriptions[video_path]
        user_prompt_template = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "video"}
                ]
            }
        ]
        user_prompt_text = self.processor.apply_chat_template(user_prompt_template, add_generation_prompt=True)
        user_prompt_outputs = self.processor(
            videos=frames,
            text=user_prompt_text,
            return_tensors="pt",
            video_metadata={"fps": fps, "total_num_frames": total_frames})
        prompt_len = user_prompt_outputs.input_ids.shape[1]
        prompt_template = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "video"}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": enriched_description}
                ]
            }
        ]
        prompt = self.processor.apply_chat_template(prompt_template)
        inputs = self.processor(
            videos=frames,
            text=prompt,
            return_tensors="pt",
            video_metadata={"fps": fps, "total_num_frames": total_frames}
        )
        item = {
            "pixel_values_videos": inputs.pixel_values_videos.squeeze(0),
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "labels": self.enriched_descriptions[video_path],
            "prompt_len": prompt_len,
            "ground_truth_keywords": self.ground_truth_labels[video_path],
            "raw_labels": self.raw_labels[video_path],
            "video_path": video_path
        }
        if "video_grid_thw" in inputs:
            item["video_grid_thw"] = inputs.video_grid_thw.squeeze(0)
        return item
    
if __name__ == "__main__":
    enrichment_csv = "src/data/enriched_classes.csv"
    video_csv = "data/test.csv"
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True)
    dataset = EnrichedDataset(
        video_csv=video_csv,
        prompt_template="Describe the medical interventions being performed on the newborn in the video.",
        processor=processor,
        num_frames=None,
        enrichment_csv=enrichment_csv
    )
    for i in range(len(dataset)):
        item = dataset[i]
        print(f"Video {i} enriched description: {item['labels']}")
        print(f"Video {i} ground truth keywords: {item['ground_truth_keywords']}")