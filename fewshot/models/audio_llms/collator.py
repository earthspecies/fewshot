import torch
import torchaudio
import transformers
from transformers import AutoTokenizer

class LanguageAudioDataCollator:
    def __init__(self, sample_rate: int, prompt_template_path: str, support_dur_sec: int, tokenizer_path: str):
        self.sample_rate = sample_rate
        self.support_dur_sec = support_dur_sec
        with open(prompt_template_path, "r") as f:
            self.prompt_template = f.read()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def format_prompt(self, timestamps):
        events_description = ', '.join([f"({start:.2f}, {end:.2f})" for start, end in timestamps])
        return self.prompt_template.replace("{timestamps}", f"[{events_description}]")

    def format_labels(self, timestamps):
        events_description = ', '.join([f"({start:.2f}, {end:.2f})" for start, end in timestamps])
        return f"[{events_description}]"

    def __call__(self, batch):
        support_audio, support_labels, query_audio, query_labels, support_timestamps, query_timestamps = zip(*batch)

        # Process prompts and labels
        formatted_prompts = [self.format_prompt(ts) for ts in support_timestamps]
        text_labels = [self.format_labels(ts) for ts in query_timestamps]

        # Process audio segments
        full_audios = [torch.cat([s_a, q_a]) for s_a, q_a in zip(support_audio, query_audio)]
        full_audios_tensor = torch.stack(full_audios)

        #TODO: finish handling tokenization here
        be_inputs = self.tokenizer(formatted_prompts, padding=True, truncation=True, return_tensors="pt")
        be_labels = self.tokenizer(text_labels, padding=True, truncation=True, return_tensors="pt")
        input_ids = be_inputs["input_ids"]
        attention_mask = be_inputs["attention_mask"]
        labels = [-100] * len(input_ids) # TODO: fix
        suffix_labels = be_labels["input_ids"]
        suffix_attention_mask = be_labels["attention_mask"]

        return {
            "audio": full_audios_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "suffix_labels": suffix_labels,
            "suffix_attention_mask": suffix_attention_mask
        }
    

