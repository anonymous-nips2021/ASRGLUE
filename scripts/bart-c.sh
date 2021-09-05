python3 -m torch.distributed.launch --nproc_per_node 4 seq2seq.py --model_name_or_path facebook/bart-large --train_file $TRAIN_FILE --validation_file $VALIDATION_FILE --test_file $TEST_FILE --text_column noisy --summary_column golden --output_dir $OUTPUT --per_device_train_batch_size=8 --per_device_eval_batch_size=16 --gradient_accumulation_steps 4 --max_steps 4000 --predict_with_generate --num_beams 4 --overwrite_output_dir --do_train --do_eval --do_predict 
