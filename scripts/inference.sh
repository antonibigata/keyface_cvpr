#!/bin/bash

# Get command line arguments with defaults
output_folder=$1
file_list=${2:-"filelist_val.txt"}
keyframes_ckpt=${3:-None}
interpolation_ckpt=${4:-None}
compute_until=${5:-45}
file_list_audio=${6:-None}

# Run the Python script with the appropriate arguments
python scripts/sampling/full_pipeline.py \
    --filelist=${file_list} \
    --filelist_audio=${file_list_audio} \
    --decoding_t 1 \
    --cond_aug 0. \
    --resize_size=512 \
    --force_uc_zero_embeddings='[cond_frames, audio_emb]' \
    --latent_folder=videos \
    --input_folder=videos \
    --model_config=scripts/sampling/configs/interpolation.yaml \
    --model_keyframes_config=scripts/sampling/configs/keyframe.yaml \
    --chunk_size=2 \
    --audio_folder=audios \
    --audio_emb_folder=audios \
    --output_folder=outputs/${output_folder} \
    --keyframes_ckpt=${keyframes_ckpt} \
    --interpolation_ckpt=${interpolation_ckpt} \
    --add_zero_flag=True \
    --extra_audio=both \
    --compute_until=${compute_until} \
    --audio_emb_type=wavlm \
    --accentuate=False \
    --emotion_states=["neutral"] \
    --recompute=True
