python train.py --dataset jhmdb-21 \
		--data_cfg cfg/jhmdb21.data \
		--cfg_file cfg/jhmdb21.cfg \
		--n_classes 21 \
		--backbone_3d resnext101 \
		--backbone_3d_weights weights/resnext-101-kinetics-hmdb51_split1.pth \
		--freeze_backbone_3d \
		--backbone_2d darknet \
		--backbone_2d_weights weights/yolo.weights \
		--freeze_backbone_2d \
		# --resume_path /usr/home/sut/yowo/backup/yowo_jhmdb-21_16f_best.pth \



