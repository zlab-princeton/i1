# i1: A Simple and Fully Open Recipe for Strong Text-to-Image Models

Official code for **i1: A Simple and Fully Open Recipe for Strong Text-to-Image Models**

> **i1: A Simple and Fully Open Recipe for Strong Text-to-Image Models**<br>
> [Boya Zeng](https://boyazeng.github.io), [Tianze Luo](https://luotianze666.github.io), [Shu Pu](https://urrealhero.github.io/MyPersonalWeb/), [Jucheng Shen](https://juchengshen.github.io), [Taiming Lu](https://taiminglu.com), [Gabriel Sarch](http://gabesarch.me), [Zhuang Liu](https://www.cs.princeton.edu/~zhuangl)
> <br>Princeton University<br>
> [[`arXiv`](https://arxiv.org/abs/2606.11289)][[`model`](https://huggingface.co/zlab-princeton/i1-3B)][[`dataset`](https://huggingface.co/datasets/zlab-princeton/i1-captions)][[`project page`](https://zlab-princeton.github.io/i1/)]

<p align="center">
<img src="./docs/static/images/teaser.png" width=90% height=90% 
class="center">
</p>

We investigate the design space of text-to-image diffusion models to understand how modeling and data choices affect model capabilities. This exploration culminates in i1, a 3B-parameter model that performs competitively with leading models at 1024-resolution, as measured by the average percentage score across GenEval, DPG-Bench, PRISM, CVTG-2K, and LongText-Bench.

## Showcase

<h3 align="center">General Image Generation</h3>

<p align="center">
<img src="./docs/static/images/showcase_general.jpg" width=90% height=90% 
class="center">
</p>

<h3 align="center">Text Rendering</h3>

<p align="center">
<img src="./docs/static/images/showcase_text_render.jpg" width=90% height=90% 
class="center">
</p>

## Open-Source Plan
We **fully open-source** the training code, data, and recipes for reproducing our i1-3B model.

 - [x] 3B Model Checkpoint \[[PyTorch](https://huggingface.co/zlab-princeton/i1-3B/blob/main/1024_resolution_checkpoint_torch.pt)\] \[[JAX](https://huggingface.co/zlab-princeton/i1-3B/blob/main/checkpoint.npz-002800000)\]
 - [x] 1B Model Checkpoint \[[PyTorch](https://huggingface.co/zlab-princeton/i1-1B/blob/main/1024_resolution_checkpoint_torch.pt)\] \[[JAX](https://huggingface.co/zlab-princeton/i1-1B/blob/main/checkpoint.npz-002800000)\]
 - [x] [JAX/TPU Training and Inference Code](jax)
 - [x] [PyTorch/GPU Inference Code](torch_inference)
 - [x] [Dataset](https://huggingface.co/datasets/zlab-princeton/i1-captions) and [Data Pipelines](data_processing)
 - [ ] JAX/GPU Training and Inference Code
 - [ ] PyTorch/GPU Training Code
 - [ ] Multi-Aspect-Ratio Checkpoint, Data Pipelines, and Training Code

## Quick Start
Install PyTorch inference environment
```bash
conda create -n i1_torch_infer python=3.11 -y
conda activate i1_torch_infer
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install numpy==1.26.4 pillow tqdm transformers==4.57.1 diffusers==0.35.1 accelerate safetensors sentencepiece
```

Generate image with your custom prompt
```bash
git clone https://github.com/zlab-princeton/i1
cd torch_inference
python generate.py \
    --prompt "Render the following text at the center of the image on a clean background: 'Flow on, river! flow with the flood-tide, and ebb with the ebb-tide! Frolic on, crested and scallop-edg'd waves!'"
```

## Code Structure

This codebase contains three independent folders.

- [data_processing](data_processing) contains the code for downloading images, recaptioning images, and creating TFRecord files for the image-caption pairs.
- [jax](jax) contains the training and inference code for our controlled experiments and the final i1-3B model in JAX.
- [torch_inference](torch_inference) contains the inference code for the final i1-3B model in PyTorch.

## Acknowledgement
We gratefully thank the Google TPU Research Cloud (TRC) program for providing the primary computing resources for this project. Additional support was provided by the Princeton Research Computing resources at Princeton University, which are managed by a consortium of groups led by the Princeton Institute for Computational Science and Engineering (PICSciE) and Research Computing.
We would like to thank Liang-Chieh Chen, Ishan Misra, Kaiming He, Yida Yin, Haozhe Chen, Wenhao Chai, Linrong Cai, Linzhan Mou, and Xingyu Fu for valuable discussions and feedback. We also thank Yufeng Xu, Shengbang Tong, Yiyang Lu, and Hanhong Zhao for helpful discussion on TPU. We are grateful to Cihang Xie's research group for sharing their JAX DiT codebase, which served as the launching point for our research.
This repository is built using the [big_vision](https://github.com/google-research/big_vision), [transformers](https://github.com/huggingface/transformers), and [diffusers](https://github.com/huggingface/diffusers) codebases.

## Citation
If you find this repository helpful, please consider citing:
```bibtex
@article{zeng2026i1,
  title={i1: A Simple and Fully Open Recipe for Strong Text-to-Image Models},
  author={Zeng, Boya and Luo, Tianze and Pu, Shu and Shen, Jucheng and Lu, Taiming and Sarch, Gabriel and Liu, Zhuang},
  journal={arXiv preprint arXiv:2606.11289},
  year={2026}
}
```