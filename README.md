
<h1 align="center">KeyFace: Expressive Audio-Driven Facial Animation for Long Sequences via KeyFrame Interpolation</h1>

<div align="center">
    <a href="https://scholar.google.com/citations?user=LuIdiV8AAAAJ" target="_blank">Antoni Bigata</a><sup>1</sup>&emsp;
    <a href="https://scholar.google.com/citations?user=ty2OYvcAAAAJ" target="_blank">Michał Stypułkowski</a><sup>2</sup>&emsp;
    <a href="https://scholar.google.com/citations?user=08YfKjcAAAAJ" target="_blank">Rodrigo Mira</a><sup>1</sup>&emsp;
    <a href="https://scholar.google.com/citations?user=zdg4dj0AAAAJ" target="_blank">Stella Bounareli</a><sup>1</sup>&emsp;
    <a href="https://scholar.google.com/citations?user=WwLpK44AAAAJ" target="_blank">Konstantinos Vougioukas</a><sup>1</sup>&emsp;
    <a href="https://scholar.google.com/citations?user=46APmkYAAAAJ" target="_blank">Zoe Landgraf</a><sup>1</sup>&emsp;
    <a href="https://scholar.google.com/citations?user=itNst7wAAAAJ" target="_blank">Nikita Drobyshev</a><sup>1</sup>&emsp;
    <a href="https://scholar.google.com/citations?user=XmOBJZYAAAAJ" target="_blank">Maciej Zieba</a><sup>3</sup>&emsp;
    <a href="https://scholar.google.com/citations?user=6v-UKEMAAAAJ" target="_blank">Stavros Petridis</a><sup>1</sup>&emsp;
    <a href="https://scholar.google.com/citations?user=ygpxbK8AAAAJ" target="_blank">Maja Pantic</a><sup>1</sup>
</div>

<div align="center">
<div class="is-size-5 publication-authors" style="margin-top: 1rem;">
          <span class="author-block"><sup>1</sup>Imperial College London,</span>
          <span class="author-block"><sup>2</sup>University of Wrocław,</span>
          <span class="author-block"><sup>3</sup>Technical University of Wroclaw</span>
</div>
</div>

<div align="center">
    <a href="https://antonibigata.github.io/KeyFace/"><img src="https://img.shields.io/badge/Project-Page-blue"></a>
    <a href="https://huggingface.co/toninio19/keyface"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow"></a>
    <a href="https://arxiv.org/abs/2503.01715"><img src="https://img.shields.io/badge/Paper-Arxiv-red"></a>
</div>

## 📋 Table of Contents
- [Abstract](#abstract)
- [Demo Examples](#demo-examples)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start Guide](#quick-start-guide)
- [Advanced Usage](#advanced-usage)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Abstract

Current audio-driven facial animation methods achieve impressive results for short videos but suffer from error accumulation and identity drift when extended to longer durations. Existing methods attempt to mitigate this through external spatial control, increasing long-term consistency but compromising the naturalness of motion. 

We propose **KeyFace**, a novel two-stage diffusion-based framework, to address these issues:
1. In the first stage, keyframes are generated at a low frame rate, conditioned on audio input and an identity frame, to capture essential facial expressions and movements over extended periods.
2. In the second stage, an interpolation model fills in the gaps between keyframes, ensuring smooth transitions and temporal coherence.

To further enhance realism, we incorporate continuous emotion representations and handle a wide range of non-speech vocalizations (NSVs), such as laughter and sighs. We also introduce two new evaluation metrics for assessing lip synchronization and NSV generation. Experimental results show that KeyFace outperforms state-of-the-art methods in generating natural, coherent facial animations over extended durations.

## Demo Examples

<div align="center">
  <table>
    <tr>
      <td>
        <video width="100%" autoplay loop muted playsinline>
          <source src="https://antonibigata.github.io/KeyFace/static/videos/front_row_1/example_1.mp4" type="video/mp4">
        </video>
      </td>
      <td>
        <video width="100%" autoplay loop muted playsinline>
          <source src="https://antonibigata.github.io/KeyFace/static/videos/front_row_1/example_3.mp4" type="video/mp4">
        </video>
      </td>
      <td>
        <video width="100%" autoplay loop muted playsinline>
          <source src="https://antonibigata.github.io/KeyFace/static/videos/front_row_1/example_5.mp4" type="video/mp4">
        </video>
      </td>
      <td>
        <video width="100%" autoplay loop muted playsinline>
          <source src="https://antonibigata.github.io/KeyFace/static/videos/front_row_1/example_4.mp4" type="video/mp4">
        </video>
      </td>
    </tr>
    <tr>
      <td>
        <video width="100%" autoplay loop muted playsinline>
          <source src="https://antonibigata.github.io/KeyFace/static/videos/front_row_2/example_1.mp4" type="video/mp4">
        </video>
      </td>
      <td>
        <video width="100%" autoplay loop muted playsinline>
          <source src="https://antonibigata.github.io/KeyFace/static/videos/front_row_2/example_2.mp4" type="video/mp4">
        </video>
      </td>
      <td>
        <video width="100%" autoplay loop muted playsinline>
          <source src="https://antonibigata.github.io/KeyFace/static/videos/front_row_2/example_5.mp4" type="video/mp4">
        </video>
      </td>
      <td>
        <video width="100%" autoplay loop muted playsinline>
          <source src="https://antonibigata.github.io/KeyFace/static/videos/front_row_2/example_4.mp4" type="video/mp4">
        </video>
      </td>
    </tr>
  </table>
</div>

## Architecture

<div align="center">
  <img src="https://antonibigata.github.io/KeyFace/static/images/more_img/drawing_obama.png" width="100%">
</div>

## Installation

### Prerequisites
- CUDA-compatible GPU
- Python 3.11
- Conda package manager

### Setup Environment

```bash
# Create conda environment with necessary dependencies
conda create -n keyface python=3.11 nvidia::cuda-nvcc conda-forge::ffmpeg -y
conda activate keyface

# Install requirements
pip install -r requirements.txt --no-deps

# Install PyTorch with CUDA support
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

### Download Pretrained Models

```bash
git lfs install
git clone https://huggingface.co/toninio19/keyface pretrained_models
```

#### Important Note on Pretrained Models

The pretrained models available in the HuggingFace repository have been retrained on non-proprietary data. As a result, the performance and visual quality of animations generated using these models may differ from those presented in the paper. 

## Quick Start Guide

### 1. Data Preparation

To use KeyFace with your own data, for simplicity organize your files as follows:
- Place video files (`.mp4`) in the `data/videos/` directory
- Place audio files (`.wav`) in the `data/audios/` directory

Otherwise you need to specify a different video_dir and audio_dir.

### 2. Running Inference

For inference you need to have the audio and video embeddings precomputed.
The simplest way to run inference on your own data is using the `infer_raw_data.sh` script which will compute those embeddings for you:

```bash
./scripts/infer_raw_data.sh \
  --video_dir="data/videos" \
  --audio_dir="data/audios" \
  --output_folder="my_animations" \
  --keyframes_ckpt="path/to/keyframes_model.ckpt" \
  --interpolation_ckpt="path/to/interpolation_model.ckpt" \
  --compute_until=45
```

This script handles the entire pipeline:
1. Extracts video embeddings
2. Computes audio embeddings (using BEATS, WavLM, and Wav2Vec2)
3. Creates a filelist for inference
4. Runs the full animation pipeline

For more control over the inference process, you can directly use the `inference.sh` script:

```bash
./scripts/inference.sh \
  output_folder_name \
  path/to/filelist.txt \
  path/to/keyframes_model.ckpt \
  path/to/interpolation_model.ckpt \
  compute_until
```

### 3. Training Your Own Models

The dataloader needs the path to all the videos you want to train on. Then you need to separate the audio and video as follows:
- root_folder:
  - videos: raw videos
  - videos_emb: embedding for your videos
  - audios: raw audios
  - audios_emb: precomputed embeddigns for the audios
  
You can have different folders but make sure to change them in the training scripts.

KeyFace uses a two-stage model approach. You can train each component separately:

#### Keyframe Model Training

```bash
./train_keyframe.sh path/to/filelist.txt [num_workers] [batch_size] [num_devices]
```

#### Interpolation Model Training

```bash
./train_interpolation.sh path/to/filelist.txt [num_workers] [batch_size] [num_devices]
```

## Advanced Usage

### Command Line Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `video_dir` | Directory with input videos | `data/videos` |
| `audio_dir` | Directory with input audio files | `data/audios` |
| `output_folder` | Where to save generated animations | - |
| `keyframes_ckpt` | Keyframe model checkpoint path | - |
| `interpolation_ckpt` | Interpolation model checkpoint path | - |
| `compute_until` | Animation length in seconds | 45 |

### Advanced Configuration

For more fine-grained control, you can edit the configuration files in the `configs/` directory.

## LipScore Evaluation

KeyFace can be evaluated using the LipScore metric available in the `evaluation/` folder. This metric measures the lip synchronization quality between generated and ground truth videos.

To use the LipScore evaluation, you'll need to install the following dependencies:

1. Face detection library: [https://github.com/hhj1897/face_detection](https://github.com/hhj1897/face_detection)
2. Face alignment library: [https://github.com/ibug-group/face_alignment](https://github.com/ibug-group/face_alignment)

Once installed, you can use the LipScore class in `evaluation/lipscore.py` to evaluate your generated animations:


## Citation

If you use KeyFace in your research, please cite our paper:

```bibtex
@misc{bigata2025keyfaceexpressiveaudiodrivenfacial,
  title={KeyFace: Expressive Audio-Driven Facial Animation for Long Sequences via KeyFrame Interpolation}, 
  author={Antoni Bigata and Michał Stypułkowski and Rodrigo Mira and Stella Bounareli and Konstantinos Vougioukas and Zoe Landgraf and Nikita Drobyshev and Maciej Zieba and Stavros Petridis and Maja Pantic},
  year={2025},
  eprint={2503.01715},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2503.01715}, 
}
```

## Acknowledgements

This project builds upon the foundation provided by [Stability AI's Generative Models](https://github.com/Stability-AI/generative-models). We thank the Stability AI team for their excellent work and for making their code publicly available.
