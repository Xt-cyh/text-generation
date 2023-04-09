# bash
labels=("positive" "negative")

for i in ${labels[@]}
do
    echo $i
    CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 train_gpt2.py --label=$i --gpus=1 \
                     --method="gpt2_ft"
    # python train_gpt2.py --label=$i --method="prompt_tuning"
done
