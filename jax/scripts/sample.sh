# adjust these based on your inference setting
ckpt_dir=/path/to/checkpoint
metric=dpg
cfg_scales=(12)
cfg_rescales=(1)
iters=(2800000)
local_save_folder=/path/to/local/folder

# update "per-proc-batch-size" based on your TPU memory and "config" based on the model you're inferencing
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --ssh-key-file=~/.ssh/google_compute_engine --project=$PROJECT_ID --zone=$ZONE --worker=all \
--command "cd i1/jax/inference && \
source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate i1_jax_train && \
hf auth login --token your_huggingface_token && \
python generate.py \
    --config configs/i1_training/1024_resolution.py \
    --prompt-type ${metric} \
    --ckpt gs://${ckpt_dir}/checkpoint.npz \
    --sample-dir ${metric}_samples \
    --per-proc-batch-size 32 \
    --image-size 1024 \
    --cfg-scale ${cfg_scales[*]} \
    --cfg-rescale ${cfg_rescales[*]} \
    --checkpoint-iters ${iters[*]}"

# helper for caculating the number of workers for the TPU machine
num_workers=$(gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --format="json(networkEndpoints)" \
    | python -c 'import json, sys; print(len(json.load(sys.stdin).get("networkEndpoints", [])))')

# loop over CFG scale values and CFG rescale values, zip the images, and copy the zipped images out to local from each worker
for cfg_scale in "${cfg_scales[@]}"; do
    for cfg_rescale in "${cfg_rescales[@]}"; do
        for iter in "${iters[@]}"; do
            padded_iter=$(printf "%09d" "$iter")
            formatted_cfg_scale=$(LC_NUMERIC=C printf "%.1f" "$cfg_scale")
            formatted_cfg_rescale=$(LC_NUMERIC=C printf "%.1f" "$cfg_rescale")
            sample_folder="${metric}_samples_${padded_iter}_cfg${formatted_cfg_scale}_rescale${formatted_cfg_rescale}"
            echo "padded_iter: $padded_iter  cfg: $formatted_cfg_scale  rescale: $formatted_cfg_rescale"
            gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --project=$PROJECT_ID --zone=$ZONE --ssh-key-file=~/.ssh/google_compute_engine --worker=all \
            --command "cd i1/jax && zip -r ${sample_folder}.zip ${sample_folder}"

            mkdir -p "${local_save_folder}_cfg${formatted_cfg_scale}_rescale${formatted_cfg_rescale}/${metric}_samples_${padded_iter}"
            for worker in $(seq 0 $((num_workers - 1))); do
                gcloud alpha compute tpus tpu-vm scp --recurse \
                "${TPU_NAME}:i1/jax/${sample_folder}.zip" \
                "${local_save_folder}_cfg${formatted_cfg_scale}_rescale${formatted_cfg_rescale}/${metric}_samples_${padded_iter}/worker${worker}.zip" \
                --project="$PROJECT_ID" \
                --zone="$ZONE" \
                --worker="$worker"
            done
        done
    done
done