from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
import torch
from torch.utils.data import Dataset
import torchaudio
import torch.nn.functional as F

import re
import torch
import torch.nn as nn
import torch.nn.functional as F


from sklearn.metrics.pairwise import cosine_similarity

import pickle
from tqdm import tqdm
import os
import numpy as np
import torch

from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AutoProcessor, ClapAudioModelWithProjection, ClapTextModelWithProjection, ClapModel
import torch.nn.functional as F
import pandas as pd

DATASET_NAME = "watkins"

EMBEDDINGS_OUTPUT_PATH = f'/home/davidrobinson/fewshot_data/data_large/{DATASET_NAME}_biolingual_embeddings.pkl'
INFO_EMBEDDINGS_OUTPUT_PATH = f'/home/davidrobinson/fewshot_data/data_large/{DATASET_NAME}_biolingual_infoembeddings.pkl'
OUTPUT_FP = f'/home/davidrobinson/fewshot_data/data_large/{DATASET_NAME}_pseudovox_with_biolingual.csv'
ANIMALSPEAK_CSV_PATH = "/home/davidrobinson/biolingual-2/csvs/release/animalspeak2_release_16k_license_dup.csv"
LOAD_EMBEDDINGS = False
WATKINS = True

class Gpt2Encoder(nn.Module):
    def __init__(self, gpt2_model, tokenizer, text_projection):
        super().__init__()
        self.gpt2_model = gpt2_model
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_projection = text_projection
    
    def forward(self, be): 

        batch_size = be['input_ids'].shape[0]
        outputs = self.gpt2_model(input_ids = be['input_ids'].to(self.device), attention_mask = be['attention_mask'].to(self.device))[0]
        indices = torch.eq(be['input_ids'], self.tokenizer.eos_token_id).long().argmax(dim=1)

        outputs = outputs[torch.arange(batch_size, device=self.device), indices]
        x = self.text_projection(outputs)
        x = F.normalize(x, dim=-1)
        return x

KEYS_TO_MODIFY_MAPPING = {
    "text_branch": "text_model",
    "audio_branch": "audio_model.audio_encoder",
    "attn": "attention.self",
    "self.proj": "output.dense",
    "attention.self_mask": "attn_mask",
    "mlp.fc1": "intermediate.dense",
    "mlp.fc2": "output.dense",
    "norm1": "layernorm_before",
    "norm2": "layernorm_after",
    "bn0": "batch_norm",
}

def rename_state_dict(state_dict, exclude_text = False):
    state_dict = {(k.replace("module.", "", 1) if k.startswith("module.") else k): v for k, v in state_dict.items()}

    model_state_dict = {}

    sequential_layers_pattern = r".*sequential.(\d+).*"
    text_projection_pattern = r".*_projection.(\d+).*"

    for key, value in state_dict.items():
        if exclude_text and "text_branch" in key:
            continue
        # check if any key needs to be modified
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        if re.match(sequential_layers_pattern, key):
            # replace sequential layers with list
            sequential_layer = re.match(sequential_layers_pattern, key).group(1)

            key = key.replace(f"sequential.{sequential_layer}.", f"layers.{int(sequential_layer)//3}.linear.")
        elif re.match(text_projection_pattern, key):
            projecton_layer = int(re.match(text_projection_pattern, key).group(1))

            # Because in CLAP they use `nn.Sequential`...
            transformers_projection_layer = 1 if projecton_layer == 0 else 2

            key = key.replace(f"_projection.{projecton_layer}.", f"_projection.linear{transformers_projection_layer}.")

        if "audio" and "qkv" in key:
            # split qkv into query key and value
            mixed_qkv = value
            qkv_dim = mixed_qkv.size(0) // 3

            query_layer = mixed_qkv[:qkv_dim]
            key_layer = mixed_qkv[qkv_dim : qkv_dim * 2]
            value_layer = mixed_qkv[qkv_dim * 2 :]

            model_state_dict[key.replace("qkv", "query")] = query_layer
            model_state_dict[key.replace("qkv", "key")] = key_layer
            model_state_dict[key.replace("qkv", "value")] = value_layer
        else:
            model_state_dict[key] = value

    return model_state_dict


def collate_str(batch):
    tensors = [item[0] for item in batch]  # Assuming the tensor is the first element
    strings = [item[1] for item in batch]  # Assuming the list of strings is the second element
    starts = [item[2] for item in batch]
    ends = [item[3] for item in batch]

    # Stack tensors
    tensors = torch.stack(tensors)

    # Strings are already in a batched list
    return tensors, strings, starts, ends


def _get_waveform_full(filename, target_sample_rate, first_channel = False):
    try:
        waveform, sample_rate = torchaudio.load(filename)
    except RuntimeError as e:
        import librosa
        waveform, sample_rate = librosa.load(filename, sr=None)
        waveform = torch.tensor(waveform).unsqueeze(0)

    if first_channel:
        waveform = waveform[0]
    else:
        waveform = torch.mean(waveform, dim=0).unsqueeze(0)

    if sample_rate != target_sample_rate:
        transform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = transform(waveform)

    return waveform

def _get_waveform(filename, start_time, end_time, target_sample_rate, max_samples, first_channel = True):
    
    waveform = _get_waveform_full(filename, target_sample_rate, first_channel)
    start_sample = int(start_time * target_sample_rate)
    end_sample = int(end_time * target_sample_rate)
    waveform = waveform[start_sample:end_sample]

    if waveform.shape[0] < max_samples:
        waveform = F.pad(waveform, (0, max_samples - waveform.shape[0]))

    return waveform, target_sample_rate


def process_file(path, chunk_duration, first_sample_only=False):
    try:
        audio_info = torchaudio.info(path)
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return []
    sr = audio_info.sample_rate
    total_samples = audio_info.num_frames
    chunk_size = sr * chunk_duration
    num_chunks = total_samples // chunk_size
    chunks = min(num_chunks, 30)
    if first_sample_only:
        num_chunks = 1

    chunks = [(path, chunk_duration * i, chunk_duration * (i + 1)) for i in range(num_chunks)]
    return chunks


class AudioDataset(Dataset):
    def __init__(self, file_paths, sample_rate, chunk_duration):
        super().__init__()
        self.file_paths = file_paths
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunks = []

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_file, path, chunk_duration) for path in self.file_paths]
            for future in tqdm(as_completed(futures), total=len(futures)):
                self.chunks.extend(future.result())

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        path, start_time, end_time = self.chunks[idx]
        waveform, _ = _get_waveform(path, start_time, end_time, self.sample_rate, self.sample_rate * self.chunk_duration)
        assert waveform.shape[0] == self.sample_rate * self.chunk_duration
        return waveform, path, start_time, end_time


class AudioEmbeddings:
    """
    Load audios from a csv
    """

    def __init__(self, model_path: str, sample_rate = 48000, biolingual_2 = True, chunk_length = 10) -> None:
        self.audio_files = []
        self.embeddings = []
        self.segment_info = []  # Store segment start and end times
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.biolingual_two = biolingual_2
        print("device", self.device)
        self.chunk_length = chunk_length
        
        self.model_audio = ClapAudioModelWithProjection.from_pretrained(model_path).to(self.device)
        
        if self.biolingual_two:
            self.model = ClapModel.from_pretrained(model_path).to(self.device)
            checkpoint = torch.load("/home/davidrobinson/biolingual-2/CLAP/models/bl1.5.pt", map_location="cpu")
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            state_dict = rename_state_dict(state_dict)
            
            self.model.load_state_dict(state_dict, strict=False)
            self.model_audio.audio_model = self.model.audio_model
            self.model_audio.audio_projection = self.model.audio_projection

            gpt_model = AutoModel.from_pretrained("/home/davidrobinson/biolingual-2/CLAP/models/gpt_clap_bl1.5.pt").to(self.device)
            self.gpt_tokenizer = AutoTokenizer.from_pretrained("/home/davidrobinson/biolingual-2/sapbert/train/models/alignment_gpt_species3")
            self.gpt_tokenizer.add_tokens("<|endoftext|>")
            self.gpt_tokenizer.add_special_tokens({"pad_token": "!"})
            self.gpt_encoder = Gpt2Encoder(gpt_model, self.gpt_tokenizer, text_projection=self.model.text_projection)

        self.model_text = ClapTextModelWithProjection.from_pretrained(model_path).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.sample_rate = sample_rate
        

    def load(self, path: str) -> None:
        """
        Load audios from a directory.

        Args:
            path: Path to the directory containing the audios.
        """
        # List all files in the directory and filter out non-audio files
        self.audio_files = []
        count = 0
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.lower().endswith(('.wav', '.mp3', '.flac')):
                    full_path = os.path.join(root, f)
                    self.audio_files.append(full_path)
                    count += 1
                    if count > 2:
                        break
                        # pass
        print("loaded audio")

    def embed_text(self, text: str):
        """
        Embed a text query using a transformer.

        Args:
            text: The text query to embed.

        Returns:
            The embedding of the text query.
        """
        # Embed the text query using the same transformer used for the audios
        if self.biolingual_two:
            be = self.gpt_tokenizer([text + ' <|endoftext|>'], return_tensors="pt", padding=True)
            text_out = self.gpt_encoder(be)
            print("text out shape", text_out.shape)

            return text_out[0].detach().cpu().numpy()

        inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
        model_outputs = self.model_text(**inputs)
        return model_outputs.text_embeds[0].detach().cpu().numpy()

    def embed_texts(self, texts: List[str], batch_size: int = 512):
        """
        Embed a list of text queries using a transformer with batching.

        Args:
            texts: The list of text queries to embed.
            batch_size: The batch size for embedding texts.

        Returns:
            A numpy array containing the embeddings of the text queries.
        """
        all_embeddings = []

        # Process the texts in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            if self.biolingual_two:
                be = self.gpt_tokenizer([text + ' <|endoftext|>' for text in batch_texts], return_tensors="pt", padding=True).to(self.device)
                text_out = self.gpt_encoder(be)
                all_embeddings.append(text_out.detach().cpu().numpy())
            else:
                inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True).to(self.device)
                model_outputs = self.model_text(**inputs)
                all_embeddings.append(model_outputs.text_embeds.detach().cpu().numpy())

        # Concatenate all the embeddings
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        return all_embeddings


    def embed_audio(self, audio):
        """
        Embed a single audio query.
        """
        # Embed the audio query using the same transformer used for the audios
        inputs = self.processor(audios=[audio], return_tensors="pt", sampling_rate=48000, padding=True).to(self.device)
        model_outputs = self.model_audio(**inputs)
        return model_outputs.audio_embeds[0].detach().cpu().numpy()
    
    def index(self) -> None:
        """
        Prepare embeddings for cosine similarity search.
        """
        # Normalize the embeddings to unit length
        self.normalized_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1)[:, np.newaxis]

    def embed_and_predict(self, pickle_path: str, file_to_embedding_pickle: str, labels_csv_path: str, prediction_output_csv: str) -> None:
        """
        Embed the audios using a transformer, make predictions based on the most similar label from the text embeddings, and save the predictions.

        Parameters:
            pickle_path (str): Path to save the full embeddings and segment info.
            file_to_embedding_pickle (str): Path to save the file-to-embedding dictionary.
            labels_csv_path (str): Path to the CSV containing possible species labels.
            prediction_output_csv (str): Path to save the predictions as a CSV file.
        """
        # Load labels from the CSV file
        labels_df = pd.read_csv(labels_csv_path)
        if WATKINS:
            labels_df = labels_df[labels_df["source"] == "Watkins"]
        species_labels = labels_df['species_common'].dropna().tolist()
        with open("/home/davidrobinson/fewshot/dataset_building/scripts/data_large/nonbio_sounds.txt", "r") as nonbio_file:
            nonbio_labels = nonbio_file.readlines()
            nonbio_labels = [label.strip().replace("\n", "") for label in nonbio_labels]
            
        
        labels = list(set(species_labels + nonbio_labels))
        print("labels", labels[2])
        print("labels", len(labels))
        # Embed species labels as text embeddings
        # species_embeddings = [self.embed_text(species) for species in species_labels]
        print("embedding texts")
        species_embeddings = self.embed_texts(labels)
        print("finished emb texts, species embeddings shape", species_embeddings.shape)

        # Normalize species embeddings for similarity comparison
        species_embeddings = np.array(species_embeddings)
        normalized_species_embeddings = species_embeddings / np.linalg.norm(species_embeddings, axis=1)[:, np.newaxis]

        # Check if the audio embeddings pickle file already exists
        if os.path.exists(pickle_path) and LOAD_EMBEDDINGS:
            with open(pickle_path, 'rb') as file:
                data = pickle.load(file)
                self.embeddings = data['embeddings']
                self.segment_info = data['segment_info']
            print("Loaded embeddings and segment info from pickle.")
        else:
            print("making dataset")
            dataset = AudioDataset(self.audio_files, sample_rate=self.sample_rate, chunk_duration=self.chunk_length)
            print("finished dataset")
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=128,
                shuffle=False,
                num_workers=3,
                pin_memory=False,
                persistent_workers=True,
                collate_fn=collate_str,
            )

            print("embedding audio")
            count = 0
            file_to_embedding = {}  # This will store the simple mapping from file path to the first embedding

            predictions = []

            for audios, files, starts, ends in tqdm(dataloader):
                count += 1
                if count > 30:
                    # break
                    pass
                    # break
                with torch.no_grad():
                    # Prepare the inputs for the model
                    x = [s.cpu().numpy() for s in audios]
                    inputs = self.processor(audios=x, return_tensors="pt", sampling_rate=48000, padding=True).to(self.device)
                    model_outputs = self.model_audio(**inputs)

                    # Get the embeddings
                    embeddings_batch = model_outputs.audio_embeds.detach().cpu().numpy()

                    # Store embeddings and corresponding segment info
                    for i in range(len(files)):
                        self.embeddings.append(embeddings_batch[i])
                        self.segment_info.append((files[i], starts[i], ends[i]))

                        # For the file-to-embedding dictionary, store only the first embedding for each file
                        if files[i] not in file_to_embedding:
                            file_to_embedding[files[i]] = embeddings_batch[i]

                        # Perform cosine similarity between the audio embedding and species embeddings
                        normalized_audio_embedding = embeddings_batch[i] / np.linalg.norm(embeddings_batch[i])
                        similarities = cosine_similarity([normalized_audio_embedding], normalized_species_embeddings).flatten()

                        # Find the most similar label
                        top_index = np.argmax(similarities)
                        top_label = labels[top_index]
                        top_similarity = similarities[top_index]

                        # Store the prediction
                        predictions.append({
                            'audio_file': files[i],
                            'start': starts[i],
                            'end': ends[i],
                            'predicted_label': top_label,
                            'similarity': top_similarity
                        })

            # Save the embeddings and segment info to the main pickle file
            with open(pickle_path, 'wb') as file:
                data = {'embeddings': self.embeddings, 'segment_info': self.segment_info}
                pickle.dump(data, file)
            print("Saved embeddings and segment info to pickle.")

            # Save the file-to-embedding dictionary to a separate pickle file
            with open(file_to_embedding_pickle, 'wb') as file:
                pickle.dump(file_to_embedding, file)
            print("Saved file-to-embedding dictionary to pickle.")

            # Convert predictions to a DataFrame and save as CSV
            predictions_df = pd.DataFrame(predictions)
            predictions_df.to_csv(prediction_output_csv, index=False)
            print(f"Saved predictions to {prediction_output_csv}")


def main():
    # Compute the audio index
    embeddings = AudioEmbeddings("davidrrobinson/BioLingual")
    # embeddings.load("/home/davidrobinson/fewshot_data/data_large/animalspeak_pseudovox_48k")
    embeddings.load("/home/davidrobinson/fewshot_data/data_large/watkins_mastertapes")
    embeddings.embed_and_predict(pickle_path=INFO_EMBEDDINGS_OUTPUT_PATH, file_to_embedding_pickle=EMBEDDINGS_OUTPUT_PATH, 
        labels_csv_path=ANIMALSPEAK_CSV_PATH, prediction_output_csv=OUTPUT_FP)
    return embeddings

if "__main__" == __name__:
    main()
