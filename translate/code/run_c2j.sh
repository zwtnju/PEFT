model=codebert
python translate.py \
  --do_train \
  --do_test \
  --local_rank -1 \
  --model_type roberta \
  --model_name_or_path microsoft/codebert-base \
  --output_dir ./saved_models/${model}/cs-java \
  --train_filename ../data/train.java-cs.txt.cs,../data/train.java-cs.txt.java \
  --dev_filename ../data/valid.java-cs.txt.cs,../data/valid.java-cs.txt.java \
  --test_filename ../data/test.java-cs.txt.cs,../data/test.java-cs.txt.java \
  --max_source_length 512 \
  --max_target_length 512 \
  --beam_size 5 \
  --train_batch_size 32 \
  --eval_batch_size 16 \
  --learning_rate 5e-5 \
  --evaluate_during_training \
  --warmup_steps 1000 \
  --train_steps 20000 \
  --eval_steps 5000  2>&1 | tee ${model}_cs2java.log

python ../evaluator/evaluator.py -ref ../data/test.java-cs.txt.java -pre ./saved_models/$model/cs-java/test.output | tee ${model}_cs2java.em
cd ../evaluator/CodeBLEU
python calc_code_bleu.py --refs ../../data/test.java-cs.txt.java --hyp ../../code/saved_models/$model/cs-java/test.output --lang java --params 0.25,0.25,0.25,0.25 | tee ${model}_cs2java.cb
cd ../../code
