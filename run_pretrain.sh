export CUDA_VISIBLE_DEVICES=0
export PYTHONIOENCODING=utf-8
export BERT_BASE_DIR=/home/horovod_bert/pretrain_model/chinese_wwm_ext_L-12_H-768_A-12
python ./horovod_bert/run_pretraining.py \
  --input_file=/tmp/tf_examples.tfrecord \
  --output_dir=/tmp/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=16 \
  --max_seq_length=512 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5
