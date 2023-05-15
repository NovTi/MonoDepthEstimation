# config template (stored as dict)

CONFIG = {
	"mode": "train",
	"train_name": "nyu_v2",  # model name
	"manual_seed": 42,
	"method": "fcn_segmentor",
	"num_epochs": 90,
	"gpu": [0],
	"warmup_epochs": 10,

	"data": {
		"dataset": "nyu",
		"data_path": "../../../dataset/nyu_v2/train_splits",
		"gt_path": "../../../dataset/nyu_v2/train_splits",
		"file_lst": "lists/nyudepthv2_train_files_with_gt.txt",
		"input_height": 416,
		"input_width": 544,
		"max_depth": 10.0,
		"mean": [0.485, 0.456, 0.406],
		"std": [0.229, 0.224, 0.225],
		"augment": ["hor_flip", "vert_flip", "resize"]
	},

	"train": {
		"batch_size": 4,
		"fix_first_conv_blocks": False,
		"fix_first_conv_block": False,
		"bn_no_track_stats": False,
		"bts_size": 512,
		"retrain": False,
		"checkpoint_path": ""
		# "save_models": True
	},

	"network": {
		'model_name': 'bts2',
		"encoder": "resnet50_bts2"
	},

	"eval": {
		"do_online_eval": True,
		"data_path_eval": "../../../dataset/nyu_v2/official_splits/test/",  # evaluate using test set?
		"gt_path_eval": "../../../dataset/nyu_v2/official_splits/test/",
		"file_lst_eval": "lists/nyudepthv2_test_files_with_gt.txt",
		"min_depth_eval": 1e-3,
		"max_depth_eval": 10.0,
		"eigen_crop": True,
		"garg_crop": False,
		"eval_freq": 5000,
	},

	"distributed": {
		"num_threads": 1,
		"world_size": 1,
		"rank": 0,
		"dist_url": '',
		"dist_backend": 'nccl',
		"gpu": [0],
		"distributed": False 
	},

	"optim": {
		"optimizer": "adamw",
		"adamw": {
			"betas": [0.9, 0.999],
			"eps": 1e-3,
			"weight_decay": 1e-2,
		},
		"sgd": {
			"weight_decay": 0.0005,
			"momentum": 0.9,
			"nesterov": False
		}
	},

	"lr": {
		"base_lr": 1.3e-4,
		"nbb_mult": 2.0,
		"metric": "iters",
		"end_lr": -1,
		"lr_policy": "lambda_poly",
		"step": {
			"gamma": 0.5,
			"step_size": 100
		}
	},

	"augment": {
		"random_rotate": True,
		"degree": 2.5,
		"do_kb_crop": False,
		"use_right": False,
		"cutmix": True
	},

	"loss": {
		"loss_type": "si_gradient",  # or silog
		"variance_focus": 0.85
	},

	"log": {
		"log_freq": 400,
		"save_freq": 500
	}
}