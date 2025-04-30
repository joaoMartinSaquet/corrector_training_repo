from dataset import dataset_handling

# ANN
from training_script.ANN import ann_training
# CGP 
from training_script.CGP import train_cgp

if __name__ == "__main__":
    # dataset_handling.read_dataset()
    ann_training.main(config_path="config/ann_config.yaml")
    # train_cgp.main(config_path="config/cgp_config.yaml")
    