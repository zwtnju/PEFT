model=codebert
python defect.py \
    --output_dir ./saved_models/${model}/ \
    --model_type roberta \
    --model_name_or_path microsoft/codebert-base \
    --do_train \
    --do_test \
    --local_rank -1 \
    --train_data_file ../data=/train.jsonl \
    --eval_data_file ../data/valid.jsonl \
    --test_data_file ../data/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --warmup_steps 100 \
    --seed 123456 2>&1 | tee ${model}.log

python ../evaluator/evaluator.py -a ../data/test.jsonl -p saved_models/${model}/predictions.txt | tee ${model}.result
