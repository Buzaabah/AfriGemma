gpus=0
DATE_WITH_TIME=$(date "+%Y%m%d-%H%M%S")

tokenizer_name=google/mt5-base
model_name=google/mt5-base
data_dir=datasets/africanlp/en-swa
max_source_length=128
max_target_length=128
val_max_target_length=75
mode=train

output_dir=results/africanlp/mt5-base
evaluation_strategy=steps
num_train_epochs=2
per_device_train_batch_size=4
per_device_eval_batch_size=4
gradient_accumulation_steps=64
eval_steps=100
save_total_limit=10
logging_steps=1000
seed=65
save_steps=1000
warmup_steps=10000
learning_rate=1e-4
logging_dir=../results/pqa/all_l3



python main_file.py\
 --tokenizer_name ${tokenizer_name} \
  --model_name ${model_name} \
  --data_dir ${data_dir} \
  --max_source_length ${max_source_length} \
  --max_target_length ${max_target_length} \
  --val_max_target_length ${val_max_target_length} \
  --mode ${mode} \
  --task_type generation_id \
  --test_type dev \
  --output_dir ${output_dir} \
  --evaluation_strategy ${evaluation_strategy} \
  --num_train_epochs ${num_train_epochs} \
  --per_device_train_batch_size ${per_device_train_batch_size} \
  --per_device_eval_batch_size ${per_device_eval_batch_size} \
  --gradient_accumulation_steps ${gradient_accumulation_steps}\
  --eval_steps ${eval_steps} \
  --save_total_limit ${save_total_limit} \
  --logging_steps ${logging_steps} \
  --seed ${seed} \
  --save_steps ${save_steps} \
  --warmup_steps ${warmup_steps} \
  --learning_rate ${learning_rate} \
  --logging_dir ${logging_dir} \
  --overwrite_output_dir \
  --load_best_model_at_end \
  --num_labels 3
#  --sharded_ddp zero_dp_3
#  --fp16