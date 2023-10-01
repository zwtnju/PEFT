model=codebert
for lang in go java javascript php python ruby
do
python search.py \
    --output_dir ./saved_models/${model}/${lang} \
    --model_type roberta \
    --model_name_or_path microsoft/codebert-base  \
    --do_train \
    --do_test \
    --train_data_file ../data/CSN/$lang/train.jsonl \
    --eval_data_file ../data/CSN/$lang/valid.jsonl \
    --test_data_file ../data/CSN/$lang/test.jsonl \
    --codebase_file ../data/CSN/$lang/codebase.jsonl \
    --cache_path ./cache/${model} \
    --num_train_epochs 5 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456 2>&1 | tee ${model}_${lang}.log
done