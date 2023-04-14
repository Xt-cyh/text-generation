# bash
alpha=(2.5 3.0)
method="DExperts"

for i in ${alpha[@]}
do
    echo $i
	label="positive"
    python eval_contrary_prompts.py --label=$label --method=$method --batch_size=5 --gpudevice='3' \
		--mode=eval --pretrained_decoder=gpt2-large --alpha=$alpha \
		--expert=/home/cyh/ctg/codes/DExperts/model/experts/sentiment/large/finetuned_gpt2_positive \
		--antiexpert=/home/cyh/ctg/codes/DExperts/model/experts/sentiment/large/finetuned_gpt2_negative
		# --expert=../DExperts/model/experts/toxicity/large/finetuned_gpt2_nontoxic \
		# --antiexpert=../DExperts/model/experts/toxicity/large/finetuned_gpt2_toxic

	label="negative"
	python eval_contrary_prompts.py --label=$label --method=$method --batch_size=5 --gpudevice='3' \
			--mode=eval --pretrained_decoder=gpt2-large --alpha=$alpha \
			--expert=/home/cyh/ctg/codes/DExperts/model/experts/sentiment/large/finetuned_gpt2_negative \
			--antiexpert=/home/cyh/ctg/codes/DExperts/model/experts/sentiment/large/finetuned_gpt2_positive
			# --expert=../DExperts/model/experts/toxicity/large/finetuned_gpt2_nontoxic \
			# --antiexpert=../DExperts/model/experts/toxicity/large/finetuned_gpt2_toxic
done
