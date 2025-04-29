from simulate_control.simulate_controller import Corrector
from pynput import mouse, keyboard
import torch
from utils import load_config




if __name__ == "__main__":

    model_type = "LSTM" # model to load
    experiment = "P0_C0" # experiment to load
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_log_path = f"../results/{experiment}/{model_type}/"
    print("logging to ", model_log_path)
    # read hyperparameters
    config = load_config("config/ann_config.yaml")
    hyperparameters = config['hyperparameters']
    hyperparameters['len_input'] = 4

    print("----------- Starting control for experiment : {} with model : {} -----------".format(experiment, model_type))
    
    # load model
    model = torch.load(model_log_path + "model.pt", weights_only=False).to("cpu")
    # model = torch.load(model_log_path + "model.pt", map_location=torch.device('cpu'))

    model = torch.jit.script(model)


    corrector = Corrector()
    corrector.set_model(model, model_type, hyperparameters)
    with mouse.Listener(on_move=corrector.on_move), keyboard.Listener(on_press=corrector.on_press) as listener:
        corrector.set_listener(listener)
        listener.join()
        print("----------- Control for experiment : {} with model : {} finished -----------".format(experiment, model_type))
        listener.stop()


