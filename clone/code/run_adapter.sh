model=codebert
for adapter_type in houlsby pfeiffer parallel prefix_tuning lora ia3
do
  python clone.py \
      --output_dir ./saved_models/${model}_adapter/${adapter_type} \
      --model_type roberta \
      --model_name_or_path microsoft/codebert-base \
      --do_train \
      --do_test \
      --local_rank -1 \
      --train_data_file ../dataset/train.txt \
      --eval_data_file ../dataset/valid.txt \
      --test_data_file ../dataset/test.txt \
      --cache_path ./cache/${model} \
      --epoch 2 \
      --block_size 400 \
      --train_batch_size 32 \
      --eval_batch_size 64 \
      --learning_rate 5e-5 \
      --max_grad_norm 1.0 \
      --evaluate_during_training \
      --warmup_steps 100 \
      --seed 123456 \
      --adapter_type ${adapter_type} \
      --do_adapter  2>&1 | tee ${model}_${adapter_type}.log

  python ../evaluator/evaluator.py -a ../dataset/test.txt -p saved_models/${model}_adapter/${adapter_type}/predictions.txt 2>&1 | tee ${model}_${adapter_type}.result
done