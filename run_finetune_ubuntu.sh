cuda=$1
epochs="15"
learning_rate="4e-5"
model_file="embindicator"
addr="0"
del_indicator_embs="0"
pretrain_model_name="mpdrg"
pretrain_path=pretrain_models/${pretrain_model_name}.pth


python myTrain.py \
    --epochs $epochs \
    --learning_rate $learning_rate \
    --pretrain_path $pretrain_path \
    --cuda $cuda \
    --model_file $model_file \
    --addr $addr \
    --del_indicator_embs $del_indicator_embs \
    --save_path ${pretrain_model_name}_save
