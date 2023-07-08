
#### training
config_file=$1
ckpt_file=$2
device_id=$3

CUDA_VISIBLE_DEVICES=${device_id} PYTHONPATH=$PYTHONPATH:. python eval_nlq.py ${config_file}  ${ckpt_file} \
${@:4}