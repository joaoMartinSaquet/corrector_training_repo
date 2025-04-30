from simulate_control.simulate_controller import Corrector
from pynput import mouse, keyboard
import torch
from utils import load_config
import pickle
import argparse




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Simulate a learned controller by chosing the model, patient and condition' \
    'line example : python main_simulate.py -model ANN -patient P0 -condition C0')
    parser.add_argument("-model", help="ANN, LSTM or CGP", default="ANN")
    parser.add_argument("-patient", help="P0 or P1", default="P0")
    parser.add_argument("-condition", help="C0 or C1", default="C0")
    args = parser.parse_args()
    print("arguments : ",args)
    model_type = args.model # model to load
    experiment = "{}_{}".format(args.patient, args.condition) # experiment to load
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_log_path = f"../results/{experiment}/{model_type}/"
    print("logging to ", model_log_path)
    # read hyperparameters
    config = load_config(f"config/{model_type.lower()}_config.yaml")
    hyperparameters = config['hyperparameters']

    print("----------- Starting control for experiment : {} with model : {} -----------".format(experiment, model_type))
    
    # load model
    model = torch.load(model_log_path + "model.pt", weights_only=False).to("cpu")
    # model = torch.load(model_log_path + "model.pt", map_location=torch.device('cpu'))

    model = torch.jit.script(model)

    # load data hyperparameters
    data_hyperparameters = pickle.load(open(model_log_path + "data_hyperparameters.p", "rb"))

    corrector = Corrector()
    corrector.set_model(model, model_type, data_hyperparameters)
    with mouse.Listener(on_move=corrector.on_move), keyboard.Listener(on_press=corrector.on_press) as listener:
        corrector.set_listener(listener)
        listener.join()
        print("----------- Control for experiment : {} with model : {} finished -----------".format(experiment, model_type))
        listener.stop()


