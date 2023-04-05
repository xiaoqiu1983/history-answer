#export BERT_BASE_DIR=/home/horovod_bert/pretrain_model/chinese_wwm_ext_L-12_H-768_A-12
export BERT_BASE_DIR=/tmp/pretraining_output/model
export DATA_DIR=/home/data
export MODEL_DIR=/home/hvd
export EXPORT_DIR=/home/hvd
export CUDA_VISIBLE_DEVICES=2
export PYTHONIOENCODING=utf-8

mpirun --allow-run-as-root \
    -np 1 \
    -H localhost:1 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^tcp \
    python run_classifier_hvd.py \
    --shots=1 \
    --loss_class=focal_loss \
    --task_name=xnli \
    --do_train=true \
    --do_eval=true \
    --do_predict=False \
    --do_export=False \
    --data_dir=$DATA_DIR/ \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=512 \
    --train_batch_size=5 \
    --learning_rate=2e-5 \
    --num_train_epochs=100.0 \
    --output_dir=$MODEL_DIR/output/ \
    --export_dir=$EXPORT_DIR/output/ |& grep -v "Read -1"
