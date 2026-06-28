# PyTorch Inference on GPUs

We provide a PyTorch reimplementation of the inference code for our i1-3B model.

## 1. Installation
```bash
conda create -n i1_torch_infer python=3.11 -y
conda activate i1_torch_infer
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install numpy==1.26.4 pillow tqdm transformers==4.57.1 diffusers==0.35.1 accelerate safetensors sentencepiece
```

## 2. Run generation

`generate.py` is a standalone file for PyTorch inference. It supports three ways to provide text-to-image prompts, as described below.

By default, we use the complex metaprompt in [metaprompt.txt](metaprompt.txt) and `Qwen/Qwen3-30B-A3B` to rewrite each input text-to-image prompt.
To reduce GPU memory requirements, you can switch to a smaller prompt rewriter by adding `--rewriter-model Qwen/Qwen3-4B-Instruct-2507`. However, generation quality may degrade.

For faster inference, you can reduce the number of inference steps with `--num-steps`, for example to `50`. You can also try switching from the default 3B model to the smaller 1B model, which was trained with the same recipe, by setting `--model-size 1B`.

### Option 1: provide input prompts with the `--prompt` flag
```bash
python generate.py \
    --prompt "Render the following text at the center of the image on a clean background: 'Flow on, river! flow with the flood-tide, and ebb with the ebb-tide! Frolic on, crested and scallop-edg'd waves!'" \
    --prompt "A 2*2 grid of photos of the same person. The upper left shows happy face, upper right shows sad face, lower left shows angry face, lower right shows exhausted face." \
    --prompt "A 2*2 grid of photos of the same person. The upper left shows exhausted face, upper right shows happy face, lower left shows sad face, lower right shows angry face."
```

### Option 2: provide a list of prompts in a `.txt` file with the `--prompts-file` flag
```bash
python generate.py --prompts-file prompt_set.txt
```

### Option 3: use a benchmark prompt set
```bash
python generate.py --prompt-set dpg
```
The `--prompt-set` options are as follows:
- `geneval`, `dpg`, `prism`, `CVTG-2K`, `longtext`: original benchmark prompt sets.
- `geneval_simple_rewrite`, `dpg_simple_rewrite`, `prism_simple_rewrite`, `CVTG-2K_simple_rewrite`, `longtext_simple_rewrite`: benchmark prompt sets rewritten by `Qwen/Qwen3-4B-Instruct-2507` using a simple metaprompt: "I have a short text-to-image prompt '{prompt}'. Please expand it into a descriptive paragraph, while making sure the generated image still clearly includes all the items mentioned in the original prompt. Please only output the rewritten prompt and nothing else."
- `geneval_complex_rewrite`, `dpg_complex_rewrite`, `prism_complex_rewrite`, `CVTG-2K_complex_rewrite`, `longtext_complex_rewrite`: benchmark prompt sets rewritten by `Qwen/Qwen3-30B-A3B` using the complex metaprompt in [metaprompt.txt](metaprompt.txt).

For prompt sets that have already been rewritten (*i.e.*, `*_simple_rewrite` and `*_complex_rewrite`), `--rewrite-prompt` should be set to `False` so that the prompts are not rewritten a second time.

`dpg`, `prism_simple_rewrite`, and `longtext` are used for evaluation in the controlled experiments. The `*_complex_rewrite` prompt sets are used for the final i1-3B evaluation in our paper.