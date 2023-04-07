# bash
label="nontoxic"
method="DExperts"
epoch=2

python eval_toxic_prompts.py --label=$label --method=$method --batch_size=5 --gpudevice='7' \
        --mode=eval --pretrained_decoder=gpt2-large \
        --expert=../DExperts/model/experts/toxicity/large/finetuned_gpt2_nontoxic \
        --antiexpert=../DExperts/model/experts/toxicity/large/finetuned_gpt2_toxic
        # --expert=../DExperts/model/experts/sentiment/large/finetuned_gpt2_positive \
        # --antiexpert=../DExperts/model/experts/sentiment/large/finetuned_gpt2_negative
