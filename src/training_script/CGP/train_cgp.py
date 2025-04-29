import sys
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
# import wandb
import pprint

from utils import *
from dataset.dataset_handling import *
from training_script.CGP.pyCGP.pycgp import cgp, evaluators, viz, cgpfunctions
from training_script.CGP.pyCGP.pycgp.cgp import *
from training_script.CGP.pyCGP.pycgp.cgpfunctions import *

OPTIMIZE_HYPERPARAMETERS = False
EXP_NAME = "P0_C0"
ANGLE = True
TRANS = False

corrector_lib0 =  [CGPFunc(f_sum, 'sum', 2, 0, '+'),
            CGPFunc(f_aminus, 'aminus', 2, 0, '-'),
            CGPFunc(f_mult, 'mult', 2, 0, '*'),
            CGPFunc(f_sin, 'sin', 1, 0, 'sin'),
            CGPFunc(f_cos, 'cos', 1, 0, 'cos'),
            CGPFunc(f_div, 'div', 2, 0, '/'),
            CGPFunc(f_const, 'c', 0, 1, 'c')
            ]   
            # CGPFunc(f_const, 'c', 0, 1, 'c')

corrector_lib1 =  [CGPFunc(f_sum, 'sum', 2, 0, '+'),
            CGPFunc(f_aminus, 'aminus', 2, 0, '-'),
            CGPFunc(f_mult, 'mult', 2, 0, '*'),
            # CGPFunc(f_max, 'div', 2, 0, '/'),
            CGPFunc(f_gt, 'gt', 2, 0, '>'),
            CGPFunc(f_log, 'log', 1, 0, 'log'),
            CGPFunc(f_sqrtxy, 'sqrtxy', 2, 0, 'sqt'),
            CGPFunc(f_const, 'c', 0, 1, 'c')
            ]


corrector_lib2 =  [CGPFunc(f_sum, 'sum', 2, 0, '+'),
            CGPFunc(f_aminus, 'aminus', 2, 0, '-'),
            CGPFunc(f_mult, 'mult', 2, 0, '*'),
            # CGPFunc(f_max, 'div', 2, 0, '/'),
            CGPFunc(f_gt, 'gt', 2, 0, '>'),
            CGPFunc(f_log, 'log', 1, 0, 'log'),
            CGPFunc(f_sqrtxy, 'sqrtxy', 2, 0, 'sqt'),
            CGPFunc(f_atan2, 'atan2', 2, 0, 'atan2'),
            CGPFunc(f_const, 'c', 0, 1, 'c')
            ]

LIBRARY = [corrector_lib0, corrector_lib1, corrector_lib2]

# wandb stuff
sweep_config = {
    'method': 'random'
    }

metric = {
    'name': 'loss',
    'goal': 'maximize'   
    }

sweep_config['metric'] = metric


parameters_dict = {
    'n_const' : {
        'values' : [0, 2, 5],
        },
    'lag' : {
        'values' : [0, 5, 30]
        },
    'trans' : {
        'values' : [True, False]
        }
    }


sweep_config['parameters'] = parameters_dict

def get_data(data_path, lag = 0, trans = False, angle = True):
    
    # prepare data
    xd, yd, _ , _= read_dataset(data_path, "dir", with_angle=angle, lag_amout=lag)

    input_name = list(xd.columns)
    if not trans:
        x = xd.to_numpy()
        y = yd.to_numpy()
    else:
        x, yt, scaler_x = preprocess_dataset(xd, yd, 'minmax', feature_range=(-1,1))
        y, xt, scaler_y = preprocess_dataset(yd, xd, 'minmax', feature_range=(-1,1))

    return x, y, input_name

def train_cgp(config = None):

    config = load_config("config/cgp_config.yaml")
    hyperparameters = config['hyperparameters']
    log_dir = f"/home/jmartinsaquet/Documents/code/IA2_codes/OfflineCorr/results/wandb/CGP"

   


    with wandb.init(config=config):
        config = wandb.config
        print("config : ", config)


        x, y, _ = get_data(f"/home/jmartinsaquet/Documents/code/IA2_codes/clone/datasets/{EXP_NAME}.csv", config.lag, config.trans, ANGLE) 
        n_input = x[0].shape[0]
        fitts_evaluator = evaluators.SREvaluator(x_train=x, y_train=y, n_inputs=n_input, n_outputs=2,
                                             col=hyperparameters['col'],
                                             row=1, library=corrector_lib2, loss='mse')
    
        # print("x.")
        # hof, hist = fitts_evaluator.evolve(mu =hyperparameters['mu'], nb_ind=hyperparameters['lambda'], num_csts=10,
        #                                 mutation_rate_nodes=config.m_rate, mutation_rate_outputs=config.m_out,
        #                                 mutation_rate_const_params=config.m_const, n_it=hyperparameters['n_gen'], folder_name=log_dir)
        hof, hist = fitts_evaluator.evolve(mu = hyperparameters['mu'], nb_ind= hyperparameters['lambda'], num_csts=config.n_const,
                                            mutation_rate_nodes=hyperparameters['m_node'], mutation_rate_outputs=hyperparameters['m_output'],
                                            mutation_rate_const_params=hyperparameters['m_const'], n_it=hyperparameters['n_gen'], folder_name=log_dir, random_genomes=True)
        
        
        data = [[x, y] for (x, y) in zip(np.arange(len(hist)), hist)]
        table = wandb.Table(data=data, columns=["x", "y"])
        wandb.log(
            {
                "my_custom_plot_id": wandb.plot.line(
                    table, "x", "y", title="test"
                )
            }
        )
        
        wandb.log({"loss": hist[-1]})



def main(config_path):

    print("---------------- {} ----------------".format(EXP_NAME))
    if OPTIMIZE_HYPERPARAMETERS:
        wandb.login()
        pprint.pprint(sweep_config)
        sweep_id = wandb.sweep(sweep_config, project="cgp_opt")
        wandb.agent(sweep_id, train_cgp, count=20)
    else:
        log_dir = f"../results/{EXP_NAME}/CGP/"
        print("logging to : ", log_dir)
        os.makedirs(log_dir, exist_ok=True)


        # load dataset
        # xd, yd, _ = read_dataset(f"/home/jmartinsaquet/Documents/code/IA2_codes/clone/datasets/{experiment_name}.csv", "vec")
        # x, y, _ = read_dataset(f"/home/jmartinsaquet/Documents/code/IA2_codes/clone/datasets/{experiment_name}.csv", "vec", 0, True)
        x, y, input_name = get_data(f"/home/jmartinsaquet/Documents/code/IA2_codes/clone/datasets/{EXP_NAME}.csv",4, TRANS, ANGLE)  

        # read hyperparameters
        config = load_config(config_path)
        hyperparameters = config['hyperparameters']
        print("hyperparameters: ")
        for key, value in hyperparameters.items():
            print(f"{key}: {value}")

        # create library
        library = corrector_lib2 #[cgp.CGPFunc()] # None for now (we can see later for customs)
        n_input = x[0].shape[0]
        fitts_evaluator = evaluators.SREvaluator(x_train=x, y_train=y, n_inputs=n_input, n_outputs=2,
                                                col=hyperparameters['col'],
                                                row=hyperparameters['row'], library=library)
        
        # print("x.")
        hof, hist = fitts_evaluator.evolve(mu = hyperparameters['mu'], nb_ind= hyperparameters['lambda'], num_csts=2,
                                            mutation_rate_nodes=hyperparameters['m_node'], mutation_rate_outputs=hyperparameters['m_output'],
                                            mutation_rate_const_params=hyperparameters['m_const'], n_it=hyperparameters['n_gen'], folder_name=log_dir, random_genomes=True)
        
        # input_name = ["x", "y", "dx", "dy"]
        output_name = ["dxe", "dye"]

        # save the best one 
        hof.save(log_dir+"hof.log")

        
        y_pred = hof.run(x).T
        eq = ''
        try : 
            eq = fitts_evaluator.best_logs(input_name, output_name)
        except: 
            pass


        # y = y.to_numpy()        
        print("y pred ! ", y_pred)
        print("equation : ", eq)
        
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(x[:, 3], x[:, 0], x[:, 1], '.', label = "input")
        ax.plot(x[:, 3], y_pred[:, 0], y_pred[:, 1], '.', label = "pred")
        ax.set_title("input with angle, dx and dy")
        ax.set_xlabel("angle")
        ax.set_ylabel("dx")
        ax.set_zlabel("dy")
        ax.legend()
        plt.figure()

        n = [i for i in range(max(x.shape))]
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(n, y[:, 0], y[:, 1], '.', label = "true")
        ax.plot(n, x[:,0], x[:,1], '.', label = "input")
        ax.plot(n, y_pred[:,0], y_pred[:,1], '.', label = "pred")
        ax.set_title("pred vs true")
        ax.set_xlabel("n")
        ax.set_ylabel("dx")
        ax.set_zlabel("dy")
        ax.legend()
        # plt.figure()
        # plt.plot(x[:,2], x[:,3], '.', label = "input")
        # plt.plot(y_pred[:,0], y_pred[:,1], '.', label = "pred")
        # plt.legend()
        
        # plt.figure()
        # plt.plot(x[:,2], x[:,3], '.', label = "input")
        # plt.plot(y[:,0], y[:,1], '.', label = "pred")
        # plt.title("input vs true")
        # plt.legend()
        
        hof_graph = hof.netx_graph(input_name, output_name, True, False, False)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        viz.draw_net(ax, hof_graph, n_input, 2)
        plt.savefig(log_dir + "graph.png")
        plt.show()

if __name__ == '__main__':



    print("---------------- {} ----------------".format(EXP_NAME))
    if OPTIMIZE_HYPERPARAMETERS:
        wandb.login()
        pprint.pprint(sweep_config)
        sweep_id = wandb.sweep(sweep_config, project="cgp_opt")
        wandb.agent(sweep_id, train_cgp, count=20)
    else:
        log_dir = f"../results/{EXP_NAME}/CGP/"
        print("logging to : ", log_dir)
        os.makedirs(log_dir, exist_ok=True)


        # load dataset
        # xd, yd, _ = read_dataset(f"/home/jmartinsaquet/Documents/code/IA2_codes/clone/datasets/{experiment_name}.csv", "vec")
        # x, y, _ = read_dataset(f"/home/jmartinsaquet/Documents/code/IA2_codes/clone/datasets/{experiment_name}.csv", "vec", 0, True)
        x, y, input_name = get_data(f"/home/jmartinsaquet/Documents/code/IA2_codes/clone/datasets/{EXP_NAME}.csv",4, TRANS, ANGLE)  

        # read hyperparameters
        config = load_config("config/cgp_config.yaml")
        hyperparameters = config['hyperparameters']
        print("hyperparameters: ")
        for key, value in hyperparameters.items():
            print(f"{key}: {value}")

        # create library
        library = corrector_lib2 #[cgp.CGPFunc()] # None for now (we can see later for customs)
        n_input = x[0].shape[0]
        fitts_evaluator = evaluators.SREvaluator(x_train=x, y_train=y, n_inputs=n_input, n_outputs=2,
                                                col=hyperparameters['col'],
                                                row=hyperparameters['row'], library=library)
        
        # print("x.")
        hof, hist = fitts_evaluator.evolve(mu = hyperparameters['mu'], nb_ind= hyperparameters['lambda'], num_csts=2,
                                            mutation_rate_nodes=hyperparameters['m_node'], mutation_rate_outputs=hyperparameters['m_output'],
                                            mutation_rate_const_params=hyperparameters['m_const'], n_it=hyperparameters['n_gen'], folder_name=log_dir, random_genomes=True)
        
        # input_name = ["x", "y", "dx", "dy"]
        output_name = ["dxe", "dye"]

        # save the best one 
        hof.save(log_dir+"hof.log")

        
        y_pred = hof.run(x).T
        eq = ''
        try : 
            eq = fitts_evaluator.best_logs(input_name, output_name)
        except: 
            pass


        # y = y.to_numpy()        
        print("y pred ! ", y_pred)
        print("equation : ", eq)
        
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(x[:, 3], x[:, 0], x[:, 1], '.', label = "input")
        ax.plot(x[:, 3], y_pred[:, 0], y_pred[:, 1], '.', label = "pred")
        ax.set_title("input with angle, dx and dy")
        ax.set_xlabel("angle")
        ax.set_ylabel("dx")
        ax.set_zlabel("dy")
        ax.legend()
        plt.figure()

        n = [i for i in range(max(x.shape))]
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(n, y[:, 0], y[:, 1], '.', label = "true")
        ax.plot(n, x[:,0], x[:,1], '.', label = "input")
        ax.plot(n, y_pred[:,0], y_pred[:,1], '.', label = "pred")
        ax.set_title("pred vs true")
        ax.set_xlabel("n")
        ax.set_ylabel("dx")
        ax.set_zlabel("dy")
        ax.legend()
        # plt.figure()
        # plt.plot(x[:,2], x[:,3], '.', label = "input")
        # plt.plot(y_pred[:,0], y_pred[:,1], '.', label = "pred")
        # plt.legend()
        
        # plt.figure()
        # plt.plot(x[:,2], x[:,3], '.', label = "input")
        # plt.plot(y[:,0], y[:,1], '.', label = "pred")
        # plt.title("input vs true")
        # plt.legend()
        
        hof_graph = hof.netx_graph(input_name, output_name, True, False, False)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        viz.draw_net(ax, hof_graph, n_input, 2)
        plt.savefig(log_dir + "graph.png")
        plt.show()