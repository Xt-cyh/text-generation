# bash
labels=("positive" "negative")
method='baseline1'

for i in ${labels[@]}
do
    echo $i
    # CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_gpt2.py --label=$i --gpus=1 \
    #                  --method="gpt2_ft" --batch_size=4 --gradient_accumulation_steps=1 --port=6668
    python train_gpt2.py --label=$i --method=$method --gpudevice=0 --batch_size=16 --gradient_accumulation_steps=4
done
