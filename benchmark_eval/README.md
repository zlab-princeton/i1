# Benchmark Evaluation

We provide instructions for evaluating images generated with the [torch_inference](../torch_inference) code. The core benchmark evaluation code is taken from the original codebases with only minimal modifications that do not influence the functionality of the code (see [MODIFICATIONS.md](MODIFICATIONS.md)).

## 1. [DPG-Bench](https://github.com/TencentQQGYLab/ELLA/tree/main/dpg_bench)

### 1.1 Environment setup
We follow [BLIP3o](https://github.com/JiuhaiChen/BLIP3o/issues/46#issuecomment-2993705076) to set up the environment.
```bash
cd dpg_bench
conda create -n dpgbench python=3.11 -y
conda activate dpgbench
conda install pytorch=2.3.0 pytorch-cuda=12.1 torchvision torchaudio --strict-channel-priority --override-channels -c https://aws-ml-conda.s3.us-west-2.amazonaws.com -c nvidia -c conda-forge -y
pip install "numpy==1.26.4" "cython==3.2.1" "setuptools==80.9.0" "wheel==0.45.1"
pip install --no-build-isolation git+https://github.com/liyaodev/fairseq.git
pip install -r requirements.txt
```

### 1.2 Download the judge model checkpoint
```bash
python cache_ckpt.py
```

### 1.3 Run evaluation
```bash
DPG_IMAGES=/path/to/generated/images
NUM_GPUs=8
RESOLUTION=256 # or 512 or 1024

# Combine the 4 images generated for each prompt into the 2*2 grid expected by the evaluation code
python process.py --root $DPG_IMAGES

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
accelerate launch --num_machines 1 --num_processes $NUM_GPUs --multi_gpu --mixed_precision "fp16" --main_process_port 29500 \
    compute_dpg_bench.py \
    --image-root-path $DPG_IMAGES/processed \
    --resolution $RESOLUTION \
    --pic-num 4 \
    --vqa-model mplug
```

## 2. [PRISM-Bench](https://github.com/rongyaofang/prism-bench)

### 2.1 Environment setup
```bash
conda create -n prism python=3.11 -y
conda activate prism
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==4.57.3" accelerate qwen-vl-utils pillow demjson3
```

### 2.2 Download the judge model checkpoint
```bash
hf download Qwen/Qwen2.5-VL-72B-Instruct \
    --local-dir /path/to/save/qwen/checkpoint
```

### 2.3 Run evaluation
```bash
cd prism-bench
PRISM_IMAGES=/path/to/generated/images

# Reorganize the generated image folder into the format expected by the evaluation code
python process.py --root $PRISM_IMAGES

python evaluation/eval_qwen25.py \
    --model_path /path/to/saved/qwen/checkpoint \
    --image_path $PRISM_IMAGES \
    --output_dir $PRISM_IMAGES/score
```

## 3. [LongText-Bench](https://github.com/X-Omni-Team/X-Omni/tree/main/textbench)

### 3.1 Environment setup
```bash
cd longtext
conda create -n longtext python=3.12 -y
conda activate longtext
pip install -r requirements.txt
pip install qwen-vl-utils
```

### 3.2 Run evaluation
```bash
LONGTEXT_IMAGES=/path/to/generated/images
NUM_GPUs=8
OUTPUT_DIR=./eval_results

# Reorganize the generated image folder into the format expected by the evaluation code
python process.py --root $LONGTEXT_IMAGES

torchrun --nnodes=1 --node-rank=0 --nproc_per_node=$NUM_GPUs --master-port 29500 \
    evaluate_text_reward.py \
    --sample_dir $LONGTEXT_IMAGES \
    --output_dir $OUTPUT_DIR \
    --mode en

cat $OUTPUT_DIR/results_chunk*.jsonl > $OUTPUT_DIR/results.jsonl
rm $OUTPUT_DIR/results_chunk*.jsonl

python3 summary_scores.py $OUTPUT_DIR/results.jsonl --mode en
```

## 4. [GenEval](https://github.com/djghosh13/geneval)

### 4.1 Environment setup
```bash
CKPT_DIR=/path/to/save/mask2former/ckpt
cd geneval
conda create -n geneval python=3.8.10
conda activate geneval
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 xformers --index-url https://download.pytorch.org/whl/cu121
pip install open-clip-torch==2.26.1 clip-benchmark openmim einops lightning "diffusers[torch]" transformers tomli platformdirs setuptools
mim install mmengine mmcv-full==1.7.2
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x
pip install -v -e .
cd ..
./evaluation/download_models.sh $CKPT_DIR
```

### 4.2 Run evaluation
```bash
GENEVAL_IMAGES=/path/to/generated/images

python evaluation/evaluate_images.py \
    $GENEVAL_IMAGES \
    --outfile $GENEVAL_IMAGES/results.jsonl \
    --model-path $CKPT_DIR

python evaluation/summary_scores.py $GENEVAL_IMAGES/results.jsonl
```

## 5. [CVTG-2K](https://github.com/NJU-PCALab/TextCrafter/tree/main/TextCrafter_Eval)

### 5.1 Environment setup
```bash
cd cvtg-2k
conda env create -f unified_environment.yml
conda activate textcrafter_eval
bash install_paddle_deps.sh
```

### 5.2 Run evaluation
```bash
CVTG_IMAGES=/path/to/generated/images

# Reorganize the generated image folder into the format expected by the evaluation code
python process.py --root $CVTG_IMAGES

python unified_metrics_eval.py --benchmark_dir prompts --result_dir $CVTG_IMAGES --output_file $CVTG_IMAGES/results.json --cache_dir /path/to/huggingface/cache --no_hf_mirror
```