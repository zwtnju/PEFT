# PEFT

## Requirements

The required Python packages for fine-tuning pre-trained models are listed in [`requirements.txt`](requirements.txt), which can be easily installed via `pip install -r requirements.txt`.

## Datasets and Fine-tuned Models
All the datasets can be downloaded from [Google Drive](https://drive.google.com/file/d/1--pIY37DciKmvjIhGmQXzJB8CuYBz00m/view?usp=sharing). 
Due to the space limit, we can not also provide our models, and all fine-tuned models for every task can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1XSIveLO4BYKtFA8Q0hjGQg?pwd=0003).


## Runs
For every task (clone, defect, search and translate), the data folder contains used dataset or the
scripts to download and pre-process datasets. The code folder contains a run.sh and run_adapter.sh or similar scripts as examples 
to fine-tune pre-trained models on corresponding tasks. All arguments are located in our paper, specific whatever you need.

For example, the [`run.sh`](clone/code/run.sh) script of clone folder is to full fine-tune pre-trained models on clone detection task,
and the [`run_adapter.sh`](clone/code/run_adapter.sh) script is to parameter-efficient fine-tune pre-trained models.


