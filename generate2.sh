# bash
labels=("positive" "negative")
method="gpt2_ft"
epoch=2

for i in ${labels[@]}
do
    echo $i
    python eval_contrary_prompts.py --label=$i --method=$method --reparameterize --batch_size=5 --gpudevice='1' \
            --model_dir=./output/$method/$i-bs-8-epoch-$epoch
done