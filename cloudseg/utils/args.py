import configargparse
import os


def parse_args():
    parser = configargparse.ArgParser(default_config_files=["default_config.yaml"])

    # Config file
    parser.add("-c", "--config", required=False, is_config_file=True, help="config file path")

    # Training parameters
    parser.add_argument("--batch_size", help="training batch size", default=8, type=int)
    parser.add_argument("--batch_size_val", help="validation batch size", default=8, type=int)
    parser.add_argument("--num_epochs", help="number of epochs", default=16, type=int)
    parser.add_argument("--num_workers", help="number of workers for dataloader", default=0, type=int)
    parser.add_argument("--model_name", help="model name for training", default="deeplab", type=str)
    parser.add_argument(
        "--pretrained",
        help="use pretrained decoder weights if supported",
        default=False,
        type=lambda x: (str(x).lower() == "true"),
    )
    parser.add_argument("--learning_rate", help="learning rate", default=0.01, type=float)
    parser.add_argument(
        "--use_clear_sky",
        help="subtract clear sky from raw ir data",
        default=True,
        type=lambda x: (str(x).lower() == "true"),
    )
    parser.add_argument(
        "--ignore_background",
        help="ignore background pixels when calculating loss and metrics",
        default=False,
        type=lambda x: (str(x).lower() == "true"),
    )
    parser.add_argument(
        "--random_mask",
        help="augment training data with random masks",
        default=True,
        type=lambda x: (str(x).lower() == "true"),
    )
    parser.add_argument(
        "--val_random_mask",
        help="augment validation and testing data with random masks",
        default=True,
        type=lambda x: (str(x).lower() == "true"),
    )
    parser.add_argument(
        "--val_rotation",
        help="augment validation set with deterministic rotations based on timestamp",
        default=True,
        type=lambda x: (str(x).lower() == "true"),
    )
    parser.add_argument(
        "--random_rotations",
        help="augment training data with random rotations",
        default=True,
        type=lambda x: (str(x).lower() == "true"),
    )
    parser.add_argument(
        "--cluster", help="training on a cluster", default=False, type=lambda x: (str(x).lower() == "true")
    )
    parser.add_argument("--experiment_name", help="name for experiment", default="default_experiment", type=str)
    parser.add_argument("--fast_dev_run", help="fast dev run with a small amount of data", action="store_true")

    # Paths
    parser.add_argument(
        "--log_dir", help="path for saving logs", default="data/training_logs", type=lambda s: os.path.abspath(s)
    )
    parser.add_argument(
        "--dataset_root",
        help="path to dataset",
        default="data/datasets/optimized_4",
        type=lambda s: os.path.abspath(s),
    )
    parser.add_argument("--gpus", help="gpus for pytorch", default=-1, type=int)
    parser.add_argument("--dataset", help="dataset for training", type=str)

    # Logging
    parser.add_argument(
        "--use_wandb",
        help="use weights and biases for logging",
        default=False,
        type=lambda x: (str(x).lower() == "true"),
    )
    parser.add_argument("--wandb_project", help="wandb project", type=str)
    parser.add_argument("--wandb_entity", help="wandb entity", type=str)

    return parser.parse_args()
