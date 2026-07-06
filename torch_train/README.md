# PyTorch Training on GPUs

This folder contains the PyTorch reimplementation of the i1 model training code.

## 1. Environment Setup
```bash
python -m venv ~/envs/i1_torch_train
source ~/envs/i1_torch_train/bin/activate
python -m pip install --upgrade pip
python -m pip install \
    "torch>=2.8" \
    "numpy==1.26.4" pillow tqdm \
    "transformers==4.57.1" "diffusers==0.35.1" accelerate safetensors sentencepiece \
    "tensorflow-cpu==2.19.0" "tensorflow-datasets==4.9.9" wandb
python -m pip install --no-deps "tensorflow-metadata==1.16.1"
```

## 2. Quick Start
**The commands below can be run immediately, without preparing data yourself.**

The commands below train a model with the following setup:

- 256-resolution GPT-Edit images (a subset of the i1 training data)
- the final i1 architecture
- the model size (*i.e.*, XL) and training length (*i.e.*, 500K steps) used in our controlled experiments

First, download the processed 256-resolution GPT-Edit TFRecords from Hugging Face (139 GB):
```bash
hf download zlab-princeton/i1-gptedit-tfrecord \
    --repo-type dataset \
    --revision main \
    --local-dir ./i1-gptedit-tfrecord
```

Then update the path to the `i1-gptedit-tfrecord` folder in [configs/quick_start.py](configs/quick_start.py):
```python
path_and_count = [
    ('/path/to/i1-gptedit-tfrecord', 1.0),
]
```

To train on 8 H100 GPUs, run:
```bash
torchrun --nproc_per_node=8 -m training.main --config configs/quick_start.py \
    --workdir /path/to/save/checkpoints \
    --fsdp 2
```

This run takes around 5.4 days.

## 3. Training with the i1 Recipe

### 3.1 Prepare the Data
Please make sure that you've followed the [data processing guide](../data_processing) to create the TFRecords. After that, please update the data path to each dataset in the config files under [configs](configs).

```python
path_and_count = [
    ('/path/to/dataset_1/tfrecord', weight_for_dataset_1),
    ('/path/to/dataset_2/tfrecord', weight_for_dataset_2),
    ('/path/to/dataset_3/tfrecord', weight_for_dataset_3),
]
```

### 3.2 Launch Training
Run the following commands sequentially to train on 256-resolution, 512-resolution, and 1024-resolution images. Replace `num_fsdp` and `num_grad_accum_steps` with values appropriate for your compute setup.

```bash
# 256-resolution training
torchrun --nproc_per_node=8 -m training.main --config configs/i1_256.py \
    --workdir /path/to/save/256_resolution_checkpoints \
    --fsdp num_fsdp --grad_accum num_grad_accum_steps

# 512-resolution training
torchrun --nproc_per_node=8 -m training.main --config configs/i1_512.py \
    --workdir /path/to/save/512_resolution_checkpoints \
    --resume /path/to/final_256_resolution_checkpoint \
    --fsdp num_fsdp --grad_accum num_grad_accum_steps

# 1024-resolution training
torchrun --nproc_per_node=8 -m training.main --config configs/i1_1024.py \
    --workdir /path/to/save/1024_resolution_checkpoints \
    --resume /path/to/final_512_resolution_checkpoint \
    --fsdp num_fsdp --grad_accum num_grad_accum_steps
```

## 4. Inference
Checkpoints saved by this PyTorch training code can be used directly with the [PyTorch inference code](../torch_inference/). To use a saved checkpoint, replace `checkpoint_path` in [generate.py](../torch_inference/generate.py):
```diff
-    checkpoint_path = hf_hub_download(
-        repo_id=MODEL_SIZE_TO_REPO_ID[args.model_size],
-        filename=f"{args.resolution}_resolution_checkpoint_torch.pt",
-        repo_type="model",
-    )
+    checkpoint_path = "/path/to/your/saved/checkpoint"
```

The training environment is compatible with the inference code, so you do not need to install a separate PyTorch inference environment.
