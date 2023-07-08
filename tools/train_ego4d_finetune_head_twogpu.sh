#### training
config_file=$1
resume_path=$2
exp_id=$3
device_id=$4
echo ${device_id}

CUDA_VISIBLE_DEVICES=${device_id} torchrun --standalone --nproc_per_node=2  train_ft.py ${config_file} \
--output ${exp_id} --resume_from_pretrain True --resume ${resume_path} \
${@:5}