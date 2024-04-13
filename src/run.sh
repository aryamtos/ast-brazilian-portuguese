model=ast
dataset=speechcommands
imagenetpretrain=True
audiosetpretrain=False
bal=none
lr=2.5e-4
epoch=20
freqm=48
timem=48
mixup=0.6
batch_size=4
fstride=10
tstride=10

dataset_mean=-4.6424894
dataset_std=4.2628665
audio_length=128
noise=True

metrics=acc
loss=CE
warmup=False
lrscheduler_start=5
lrscheduler_step=1
lrscheduler_decay=0.85



# if [ -d $exp_dir ]; then
#   echo 'exp exist'
#   exitf
# fi
# mkdir -p $exp_dir

tr_data=metadados_train_spotify_labels_binary.json
val_data=metadados_train_spotify_labels_binary.json
exp_dir=est-speechcommands-f10-t10-pTrue-b128-lr2.5e-4


python3 run.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${val_data} --exp-dir $exp_dir \
--label-csv /egs/spotify_dataset/data/class_labels_indices.csv --n_class 2 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise} \
--metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
--tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain --audioset_pretrain $audiosetpretrain > $exp_dir/log.txt