cd gector/
python train.py --train_set TRAIN_SET --dev_set DEV_SET --model_dir MODEL_DIR
python predict.py --model_path MODEL_PATH --vocab_path VOCAB_PATH --input_file INPUT_FILE --output_file OUTPUT_FILE
