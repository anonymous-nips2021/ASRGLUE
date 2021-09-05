python augmentation/train.py --model_name_or_path $GPT_MODEL --init_checkpoint $GPT_MODEL/pytorch_model.bin --train_input_file $TRAIN_FILE --eval_input_file $EVAL_FILE --output_dir $OUTPUT --seed 42 --max_seq_length 128 --train_batch_size 512 --gradient_accumulation_steps 8 --eval_batch_size 64 --learning_rate 1e-5 --valid_step 5000 --warmup_proportion 0.4 --fp16 false --train_epoch $EPOCH_NUM

python augmentation/src/gen.py --path $MODEL --input $INPUT_FILE –-output $OUTPUT_FILE
