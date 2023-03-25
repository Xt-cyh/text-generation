# bash
labels=positive
method="DExperts"
epoch=2

python eval_contrary_prompts.py --label=$i --method=$method --batch_size=10
