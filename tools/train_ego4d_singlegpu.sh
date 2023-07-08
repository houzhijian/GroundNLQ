
#### training
config_file=$1
exp_id=$2
device_id=$3

CUDA_VISIBLE_DEVICES=${device_id} torchrun --standalone --nproc_per_node=1  train.py ${config_file} \
--output ${exp_id} \
${@:4}