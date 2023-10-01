for adapter_type in houlsby pfeiffer parallel prefix_tuning lora ia3
  do
    python calc_param.py \
        --model_type t5 \
        --model_name_or_path t5-large \
        --adapter_type ${adapter_type} \
        --do_adapter 2>&1 | tee test.log

  done