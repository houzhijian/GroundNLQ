
#### training
config_file=$1
exp_id=$2

torchrun --standalone --nproc_per_node=4 train.py ${config_file} \
--output ${exp_id} \
${@:3}