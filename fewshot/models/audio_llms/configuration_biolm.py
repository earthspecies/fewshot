from transformers import PretrainedConfig, GemmaConfig

class BioLMConfig(PretrainedConfig):
    def __init__(self, sample_rate = 16000, aves = True, audio_chunk_size_sec = 2, gemma_config = None, **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.audio_chunk_size_sec = audio_chunk_size_sec
        self.aves = aves
        if gemma_config is None:
            gemma_config = {}
        self.gemma_config = GemmaConfig(gemma_config)
    