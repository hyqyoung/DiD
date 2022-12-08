nohup python DP_run.py \
		--gpus '0,1,2,3,4,5,6,7' \
		--arch ViT-B_16 \
		--train_steps 60000 \
		--b 32 \
		--lr 5e-2 \
		--max_clip_grad_norm 5.0 \
		--weight-decay 1e-5 \
		--data COCO2014 \
		--data_root_dir 'Your path to Dataset' \
		-i 448 \
		--note 'Your experiments note' >/dev/null 2>log &
		