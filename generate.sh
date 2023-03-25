# bash
# labels=("world" "sport" "business" "sci")
labels=("positive" "negative")
method="prompt_tuning"
epoch=2

for i in ${labels[@]}
do
    echo $i
    python generate.py --label=$i --method=$method --reparameterize --batch_size=5 \
            --model_dir=./output/$method/$i-mlp-prefixlen-10-bs-4-epoch-$epoch.pth
    #python eval_contrary_prompts.py --label=$i --method=$method --reparameterize --batch_size=10 \
    #        --model_dir=./output/$method/$i-mlp-prefixlen-10-bs-4-epoch-$epoch.pth
done