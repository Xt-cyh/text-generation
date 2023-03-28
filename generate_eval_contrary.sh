# bash
label="positive"
method="DExperts"
epoch=2

python eval_contrary_prompts.py --label=$label --method=$method --batch_size=5 --gpudevice='5' \
        --pretrained_decoder=gpt2-large \
        --expert=../DExperts/model/experts/sentiment/large/finetuned_gpt2_positive \
        --antiexpert=../DExperts/model/experts/sentiment/large/finetuned_gpt2_negative
