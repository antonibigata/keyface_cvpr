filelist=$1
workers=${2:-6}
batch_size=${3:-1}
device=${4:-1}
python main.py --base configs/example_training/interpolation/interpolation_base.yaml --wandb True lightning.trainer.num_nodes 1 \
    lightning.strategy=deepspeed_stage_1 lightning.trainer.precision=32 model.base_learning_rate=1.e-5 \
    data.params.train.datapipeline.filelist=$filelist \
    data.params.train.datapipeline.video_folder=videos  \
    data.params.train.datapipeline.audio_folder=audios \
    data.params.train.datapipeline.audio_emb_folder=audios_emb \
    data.params.train.datapipeline.latent_folder=videos_emb \
    data.params.train.loader.num_workers=$workers \
    data.params.train.datapipeline.audio_in_video=False \
    data.params.train.datapipeline.load_all_possible_indexes=False \
    lightning.trainer.devices=$device lightning.trainer.accumulate_grad_batches=1 \
    model.params.network_config.params.audio_cond_method=both_keyframes \
    'model.params.to_freeze=["time_"]' 'model.params.to_unfreeze=["time_embed"]' \
    data.params.train.loader.batch_size=$batch_size \
    model.params.loss_fn_config.params.lambda_lower=2. data.params.train.datapipeline.add_extra_audio_emb=True \