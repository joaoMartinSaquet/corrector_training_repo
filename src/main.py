from dataset import dataset_handling
from training_script.ANN import ann_training


if __name__ == "__main__":
    # dataset_handling.read_dataset()
    ann_training.main(config_path="src/config/ann_config.yaml")
    