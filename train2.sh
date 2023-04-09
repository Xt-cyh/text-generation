# bash
labels=("positive" "negative")

for i in ${labels[@]}
do
    echo $i
    #CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_gpt2.py --label=$i --gpus=1 \
    #                 --method="prompt_tuning" --batch_size=2  --gradient_accumulation_steps=4 --port=6667
    python train_gpt2.py --label=$i --method="prompt_tuning" --reparameterize --gpudevice=3  --batch_size=2 --gradient_accumulation_steps=4
done
