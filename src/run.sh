DEVICE=cuda
CUDA=0
EXP_NAME=[our_exp_name]
CONFIG_LIST=[customized_config_files]
MODEL=daksr

for DATA in ml-100k last-fm amazon-book;
do
    CMD="python main.py --mode tune --model=${MODEL} --config_list=${CONFIG_LIST} --dataset=${DATA} --logname=${EXP_NAME} --device=${DEVICE} --cuda=${CUDA}"
    echo ${CMD}
    ${CMD}
done
