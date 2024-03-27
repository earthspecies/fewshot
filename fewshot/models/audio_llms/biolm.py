from typing import Optional
import torch
import json
from fewshot.models.audio_llms.adapter import AdapterNetwork
from fewshot.models.audio_llms.aves_encoder import AvesEncoder
from transformers import PreTrainedModel, GemmaConfig, GemmaForCausalLM
from torch import nn



class BioLMhf(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.gemma_config = GemmaConfig(config.gemma_config)
        self.gemma_model = GemmaForCausalLM(self.gemma_config)

class BioLM(nn.Module):
    def __init__(self, aves_args, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gemma_model = GemmaForCausalLM("google/gemma-7b-it")
        self.audio_encoder = AvesEncoder(aves_args)
        self.aves_args = aves_args
        self.adapter = AdapterNetwork(aves_args)

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None,
                audio_values: Optional[torch.FloatTensor] = None,
                audio_attention_mask: Optional[torch.LongTensor] = None,
                suffix_input_ids: Optional[torch.LongTensor] = None,
                suffix_attention_mask: Optional[torch.LongTensor] = None,
                suffix_labels: Optional[torch.LongTensor] = None
    ):
        #1. Get audio embeddings
        audio_embeds = self.audio_encoder(audio_values)
        audio_embeds = self.adapter(audio_embeds, audio_embeds)
        audio_labels = torch.LongTensor(audio_embeds.size(0), audio_embeds.size(1)).fill_(-100).to(audio_embeds.device) # don't train on audio tokens

        #2. LLM forward
        prefix_embeds = self.gemma_model.get_input_embeddings()(input_ids)
        suffix_embeds = self.gemma_model.get_input_embeddings()(suffix_input_ids)

        inputs_embeds = torch.cat([prefix_embeds, audio_embeds, suffix_embeds], dim=1)
        attention_mask = torch.cat([attention_mask, audio_attention_mask, suffix_attention_mask], dim=1)
        labels = torch.cat([labels, audio_labels, suffix_labels], dim=1)

        return self.gemma_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=labels,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        suffix_input_ids,
        audio_values=None,
        generation_config=None
    ):
        inputs_embeds, attention_mask = [], []

        prefix_embeds = self.gemma_model.get_input_embeddings()(input_ids)
        prefix_attns = torch.ones(prefix_embeds.size(0), prefix_embeds.size(1), dtype=torch.long).to(prefix_embeds.device)
        inputs_embeds.append(prefix_embeds)
        attention_mask.append(prefix_attns)

        if audio_values is not None:
            audio_embeds = self.audio_encoder(audio_values)
            inputs_embeds.append(audio_embeds)
            audio_attention_mask = torch.ones(audio_embeds.size(0), audio_embeds.size(1), dtype=torch.long).to(audio_embeds.device)
            attention_mask.append(audio_attention_mask)


        suffix_embeds = self.gemma_model.get_input_embeddings()(suffix_input_ids)
        suffix_attns = torch.ones(suffix_embeds.size(0), suffix_embeds.size(1), dtype=torch.long).to(suffix_embeds.device)
        inputs_embeds.append(suffix_embeds)
        attention_mask.append(suffix_attns)

        inputs_embeds = torch.cat(inputs_embeds, dim=1)
        attention_mask = torch.cat(attention_mask, dim=1)

        return self.gemma_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config
        )
    
    @torch.no_grad()
    def chat(
        self,
        history,
        generation_config=None
    ):
        inputs_embeds = []

        for h in history:
            if len(h) == 1:
                ### text
                input_ids = h[0]
                embeds = self.gemma_model.get_input_embeddings()(input_ids)
                inputs_embeds.append(embeds)
            elif len(h) == 2:
                ### speech
                speech_values, speech_attention_mask = h[0], h[1]
                speech_embeds, _ = self.get_speech_features(speech_values, speech_attention_mask)
                inputs_embeds.append(speech_embeds)
            else:
                raise NotImplementedError
        
        inputs_embeds = torch.cat(inputs_embeds, dim=1)

        return self.gemma_model.generate(
            inputs_embeds=inputs_embeds,
            generation_config=generation_config
        )
