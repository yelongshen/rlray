
#export WANDB_PROJECT="s2lm-speech"
#data_config=configs/phi3_audio/cascade_encoder_stage1/fast-llm-speech/phi4-7b-9L-fast-llm-s1-2505-data-debug.yaml
#training_config=configs/phi3_audio/cascade_encoder_stage1/fast-llm-speech/phi4-7b-9L-fast-llm-s1-p1.yaml
#local_tmp_dir=/tmp/training_sync/
#local_output_dir=/tmp/training_output/$wandb_exp_run_name/
#remote_output_dir=$orange_model_storage/projects/ruchaofan/amlt-results/$wandb_exp_run_name/

#echo "----------Syncing remote checkpoint dir to local for potential resumed training-----------"
#bbb sync -q --concurrency 128 $remote_output_dir $local_output_dir
#[ -d $local_output_dir ] || mkdir -p $local_output_dir

cd "$(dirname "$0")"
echo "-------------Start Training--------------"
node0_ip=`brix pods "$BRIX_POOL-0" | awk 'NR>1 {print $9}'`
MASTER_ADDR="${node0_ip}:12345"
nnodes=`brix pods | grep "$BRIX_POOL" | wc -l`
logdir=$RCALL_LOGDIR
echo "MASTER_ADDR: $MASTER_ADDR with nnodes: $nnodes"

torchrun --nnodes=${nnodes} --nproc-per-node=$BRIX_GPU_REQUEST --node-rank=$RCALL_INSTANCE_INDEX --rdzv-endpoint=${MASTER_ADDR} \
    mini_rl_example.py 