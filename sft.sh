export PROC_PER_NODE=1
export NODE_COUNT=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=5680  # change your port 

#########################################################
## support single machine and multi-gpu training
#########################################################

torchrun \
 --nproc-per-node $PROC_PER_NODE \
 --master_addr $MASTER_ADDR \
 --master_port ${MASTER_PORT:-5678} \
 --nnodes $NODE_COUNT \
 --node_rank $NODE_RANK \
 ./main_sft.py config_path=./config.yaml

