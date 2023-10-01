model=codebert
for adapter_type in houlsby pfeiffer parallel prefix_tuning lora ia3
do
  python defect.py \
      --output_dir ./saved_models/${model}_adapter/${adapter_type} \
      --model_type roberta \
      --model_name_or_path microsoft/codebert-base \
      --do_train \
      --do_test \
      --local_rank -1 \
      --train_data_file ../dataset/train.jsonl \
      --eval_data_file ../dataset/valid.jsonl \
      --test_data_file ../dataset/test.jsonl \
      --epoch 5 \
      --block_size 400 \
      --train_batch_size 16 \
      --eval_batch_size 4 \
      --learning_rate 2e-5 \
      --max_grad_norm 1.0 \
      --evaluate_during_training \
      --warmup_steps 100 \
      --seed 123456 \
      --adapter_type ${adapter_type} \
      --do_adapter  2>&1 | tee ${model}_${adapter_type}.log

  python ../evaluator/evaluator.py -a ../dataset/test.jsonl -p saved_models/${model}_adapter/${adapter_type}/predictions.txt 2>&1 | tee ${model}_${adapter_type}.result
done