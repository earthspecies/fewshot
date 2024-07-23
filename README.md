# ESP Fewshot

This is the repository connected to the DCASE 2024 Few-shot Bioacoustic Event Detection Challenge. Our technical report can be found here (link to come).

## Example setup

Create conda environment: `conda create --name=fewshot pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia`

Install requirements: `pip install -r requirements.txt`

Install this package: `pip install -e .`

Get trained model weights: `mkdir weights; wget https://storage.googleapis.com/esp-public-files/fewshot/atst-finetuned-40s-support-windowed.pt weights; wget https://storage.googleapis.com/esp-public-files/fewshot/atstframe_base.ckpt weights`

## Example usage

Get example data: `wget wget https://storage.googleapis.com/esp-public-files/fewshot/example.zip ./, unzip example.zip`

Then, see `example_usage.py`