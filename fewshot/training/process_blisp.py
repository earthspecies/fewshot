import json
import torch
import torchaudio
import numpy as np
import os
import yaml


from fewshot.models.fewshot_model import FewShotAudioLLM
from fewshot.data.data import get_dataloader, FewshotDataset
from fewshot.dataclasses import load_config_from_yaml


class FewShotAudioLLMData:
    def __init__(self, sample_rate: int, data_args):
        self.sample_rate = sample_rate
        with open("fewshot/models/prompts/zeroshot.prompt", "r") as f:
            prompt_template = f.read()
        self.prompt_template = prompt_template  # Template for the prompt including placeholders for timestamps
        self.data_args = data_args

    def format_prompt(self, support_labels: torch.Tensor) -> str:
        # Convert support_labels to timestamps
        support_stamps = self.get_event_timestamps(support_labels, 0)
        # Format timestamps into the specified format
        events_description = ', '.join([f"({start:.2f}, {end:.2f})" for start, end in support_stamps])
        # Insert into the prompt template
        formatted_prompt = self.prompt_template.replace("{timestamps}", f"[{events_description}]")
        return formatted_prompt

    def format_labels(self, query_labels: torch.Tensor, offset: float) -> str:
        # Convert support_labels to timestamps
        query_stamps = self.get_event_timestamps(query_labels, offset)
        events_description = ', '.join([f"({start:.2f}, {end:.2f})" for start, end in query_stamps])
        assert isinstance(events_description, str), "events_description should be a string"
        return events_description

    def get_event_timestamps(self, labels: torch.Tensor, offset: float) -> list:
        stamps = []
        start = None
        labels = labels.cpu().numpy()  # Convert to CPU numpy array for iteration
        for i, label in enumerate(labels):
            if label == 2 and start is None:
                start = i
            elif (label == 0 or label == 1) and start is not None:
                end = i
                stamps.append((start / self.sample_rate + offset, end / self.sample_rate + offset))
                start = None
        if start is not None:
            stamps.append((start / self.sample_rate + offset, len(labels) / self.sample_rate + offset))
        return stamps

    def make_dataset(self, dataloader, prefix = ""):
        # Iterate over the dataset
        output_path = f"{prefix}_fewshot_instructions.jsonl"

        #save audios as we go

        i = 0
        for i, (audio, labels, label_mask) in enumerate(dataloader):
            # 1. get support and query audio
            support_audio = audio[:, :self.data_args.support_dur_sec*self.data_args.sr]
            query_audio = audio[:, self.data_args.support_dur_sec*self.data_args.sr:]
            
            # 2. get support and query labels
            support_labels = labels[:, :self.data_args.support_dur_sec*self.data_args.sr]
            query_labels = labels[:, self.data_args.support_dur_sec*self.data_args.sr:]
            # Convert audio tensors to file paths

            offset = self.data_args.support_dur_sec
            # Format support labels
            formatted_prompts = [self.format_prompt(support_label) for support_label in support_labels]
            text_labels = [self.format_labels(query_label, offset) for query_label in query_labels]

            print("text labels", text_labels)

            # Concatenate support and query audio tensors
            full_audios = torch.cat([support_audio, query_audio], dim=1)  # Ensure this concatenation is correct based on your tensor shapes
            # print("query_audios shape", query_audio.shape)
            # Convert to numpy array
            # Convert to list of file paths
            audio_paths = []
            for audio in full_audios:
                # audio = audio.numpy()
                audio_path = f"/home/davidrobinson/fewshot/fewshot_demo_clips/{prefix}_audio_{i}.wav"
                # print("audio shape", audio.shape)
                torchaudio.save(audio_path, audio.unsqueeze(0), self.sample_rate)
                audio_paths.append(audio_path)
                i += 1


            # Convert audio file paths to list format expected by the tokenizer
            combined_prompts = [{'audio': audio_path, "instruction": formatted_prompt, "text": text_label} for audio_path, formatted_prompt, text_label in zip(audio_paths, formatted_prompts, text_labels)]

            # Write to JSONL
            with open(output_path, "a") as f:
                for prompt in combined_prompts:
                    f.write(json.dumps(prompt) + "\n")

if __name__ == "__main__":
    
    import sys
    
    args = sys.argv[1:]
    
    # set output dir
    
    output_dir = 'fewshot_demo_clips'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the configuration
    args = load_config_from_yaml('mllm.yaml')
    
    dataloader_train, dataloader_eval = get_dataloader(args.data_args, shuffle = False)
    fewshot = FewShotAudioLLMData(16000,  args.data_args)
    fewshot.make_dataset(dataloader_train)
    fewshot.make_dataset(dataloader_eval, "eval")