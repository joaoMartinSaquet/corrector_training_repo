from dataset import dataset_handling
import argparse


# ANN
from training_script.ANN import ann_training
# CGP 
from training_script.CGP import train_cgp



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-model", help="ANN, LSTM or CGP", default="ANN")
 
    args = parser.parse_args()

    if args.model == "ANN":
        ann_training.main(config_path="config/ann_config.yaml")
    elif args.model == "LSTM":
        ann_training.main(config_path="config/ann_config.yaml")
    elif args.model == "CGP":
        train_cgp.main(config_path="config/cgp_config.yaml")
    else:
        raise ValueError(f"Unknown model: {args.model}")
    