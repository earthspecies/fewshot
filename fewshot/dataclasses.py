from dataclasses import dataclass
import yaml

@dataclass
class DataArgs:
    background_audio_dir: str
    pseudovox_audio_dir: str
    background_audio_info_fp: str
    pseudovox_info_fp: str
    min_background_duration: float
    max_pseudovox_duration: float
    min_cluster_size: int
    birdnet_confidence_strict_lower_bound: float
    cluster_column: str
    sr: int
    batch_size: int
    num_workers: int
    n_synthetic_examples: int
    support_dur_sec: float
    query_dur_sec: float

@dataclass
class TrainingArgs:
    learning_rate: float
    epochs: int
    optimizer: str

@dataclass
class ModelArgs:
    model_type: str
    model_name: str
    # Add more model-specific fields here

@dataclass
class Config:
    data_args: DataArgs
    training_args: TrainingArgs
    model_args: ModelArgs

def load_config_from_yaml(yaml_file_path: str) -> Config:
    with open(yaml_file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
        
    return Config(
        data_args=DataArgs(**yaml_data['data_args']),
        training_args=TrainingArgs(**yaml_data['training_args']),
        model_args=ModelArgs(**yaml_data['model_args'])
    )
