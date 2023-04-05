python ./horovod_bert/create_pretraining_data.py \
	--input_file=./sample_text.txt \
	--output_file=/tmp/tf_examples.tfrecord \
	--vocab_file=/home/horovod_bert/pretrain_model/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt \
	--do_lower_case=True \
	--max_seq_length=512 \
	--max_predictions_per_seq=20 \
	--masked_lm_prob=0.15 \
	--random_seed=12345 \
	--dupe_factor=5
