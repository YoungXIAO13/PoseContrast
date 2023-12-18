export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export NCCL_NET_PLUGIN=none
export NCCL_IB_TIMEOUT=22
export GLOO_SOCKET_IFNAME=bond0
export NCCL_IB_QPS_PER_CONNECTION=32
export CUDA_DEVICE_MAX_CONNECTIONS=1


MODEL_SIZE="65B_BASE_64"
MODEL_LOAD=$MODEL_SIZE
TRAIN_SN=24414062
MUL=1
TRAIN_SN=$[TRAIN_SN*$MUL]

GA_STEP=32
TENSORPN=4
PIPEPN=8
VP=2
MICRO_BS=1
LAYER_NUM=80
HIDDEN_SIZE=8192
HEAD_NUM=64


GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}
NNODES=${WORLD_SIZE}
GLOBAL_GPU_NUM=$(($GPUS_PER_NODE*$NNODES))
LR_DECAY_SN=$((${TRAIN_SN}))
LR=0.00015
MINLR=0.000015
DROPOUT=0.0
SEQ_LENGTH=4096

TENSOR_PARALLEL=4
PIPELINE_PARALLEL=8

# HEAD_DIM=128 #max value is 128 when using flash-attn

GLOBAL_BS=1024
WARMUPSN=$((${GLOBAL_BS}*2000))
ITERSPEED100=143 #seconds per 100 iters.
SAVE_INTERVAL=1000 #save onece in each 2 hours
#SAVE_INTERVAL=$(($TRAIN_SN/$GLOBAL_BS/$MUL/${SAVENUM}+1))
EVAL_INTERVAL=1000000


# Change for multinode config

cd /cpfs/2926428ee2463e44/user/zhuran/huawei-baseline/agi-megatron-lm

CHECKPOINT_PATH="/cpfs/2926428ee2463e44/user/zhuran/65b-ckpt/"

VOCAB_FILE="/cpfs/2926428ee2463e44/user/zhuran/vocab-hw/w80k_split"


DATA_PATH="/cpfs/2926428ee2463e44/user/zhuran/pile/pile_v4_w80k_split"
VALDATASET=$DATA_PATH
TESTDATASET=$VALDATASET


TENSORBOARD_DIR=${CHECKPOINT_PATH}
TENSORBOARD_DIR=$TENSORBOARD_DIR/$MODEL_SIZE

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes ${NNODES} \
    --node_rank ${RANK} \
    --master_addr ${MASTER_ADDR}  \
    --master_port ${MASTER_PORT}  \
"

GPT_ARGS="
    --use-distributed-optimizer \
    --tensor-model-parallel-size ${TENSOR_PARALLEL} \
    --pipeline-model-parallel-size ${PIPELINE_PARALLEL} \
    --sequence-parallel \
    --num-layers ${LAYER_NUM} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${HEAD_NUM} \
    --attention-dropout ${DROPOUT} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --hidden-dropout ${DROPOUT} \
    --seq-length ${SEQ_LENGTH} \
    --micro-batch-size ${MICRO_BS} \
    --global-batch-size ${GLOBAL_BS} \
    --lr ${LR} \
    --train-samples ${TRAIN_SN} \
    --lr-decay-samples ${LR_DECAY_SN} \
    --lr-decay-style cosine \
    --min-lr ${MINLR} \
    --weight-decay 0.1 \
    --lr-warmup-samples ${WARMUPSN} \
    --clip-grad 1.0 \
    --bf16 \
    --eod-mask-loss \
    --reset-position-ids \
    --reset-attention-mask \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-08 \
    --use-flash-attn \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --timing-log-level 0 \
    --normalization RMSNorm \
    --transformer-impl transformer_engine \
"

GPU_ARGS="
    --overlap-p2p-communication \
     --num-layers-per-virtual-pipeline-stage ${VP} \
"

DATA_ARGS="
    --train-data-path $DATA_PATH \
    --valid-data-path $TESTDATASET \
    --test-data-path $TESTDATASET \
    --tokenizer-type SentencePieceTokenizer \
    --vocab-file $VOCAB_FILE \
    --tokenizer-model $VOCAB_FILE \
    --data-impl mmap \
    --data-cache-path $CHECKPOINT_PATH/cache
"

OUTPUT_ARGS="
    --log-timers-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --log-interval 1 \
    --save-interval ${SAVE_INTERVAL} \
    --eval-interval ${EVAL_INTERVAL} \
    --eval-iters 10
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $GPU_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH/$MODEL_SIZE \
    --load $CHECKPOINT_PATH/$MODEL_LOAD
