
# Fed-CCVR configuration file
conf = {

	# Data type: tabular, image
	"data_type" : "tabular",

	# Model selection: mlp, cnn, resnet20
	"model_name" : "resnet20",

	# Processing method: fed_ccvr
	"no-iid": "fed_ccvr",

	# Global epochs
	"global_epochs" : 10,

	# Local epochs
	"local_epochs" : 3,

	# Dirichlet parameter
	"beta" : 0.5,

	"batch_size" : 128,

	"weight_decay": 0.0001,

    # Learning rate
	"lr" : 0.1,

	"momentum" : 0.9,

	# Number of classes
	"num_classes": 100,

	# Number of parties/nodes
	"num_parties": 8,

    # Model aggregation weight initialization
	"is_init_avg": True,

    # Local validation set split ratio
	"split_ratio": 0.3,

    # Label column name
	"label_column": "label",

	# Data column name
	"data_column": "file",

    # Test data
	"test_dataset": "./data/cifar10/test/test.csv",

    # Training data
	"train_dataset" : "./data/cifar10/train/train.csv",

    # Model save directory
	"model_dir":"./save_model/",

    # Model filename
	"model_file":"model.pth",

	"retrain":{
		"epoch": 10,
		"lr": 0.0001,
		"num_vr":500
	}
}