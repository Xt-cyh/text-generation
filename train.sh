# bash
labels=("world" "sport" "business" "sci")
# labels=("pos" "neg")

for i in ${labels[@]}
do
    echo $i
    # CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 train_gpt2.py --label=$i --gpus=1 --method="prompt_tuning"
    python train_gpt2.py --label=$i --method="prompt_tuning"
done
