import numpy as np
from aux_classes_and_fcns import*
from network_fcns import*
from algorithms import*
import random
import pickle
import argparse

####################################################################################
np.random.seed(3)
random.seed(3)

SiouxNetwork, SiouxNetwork_data = Create_Network()
OD_demands = pandas.read_csv("SiouxFallsNet/SiouxFalls_OD_matrix.txt", header=None)
OD_demands = OD_demands.values
Strategy_vectors, OD_pairs = Compute_Strategy_vectors(OD_demands, SiouxNetwork_data.Freeflowtimes, SiouxNetwork, SiouxNetwork_data.Edges)
N = len(Strategy_vectors)   # number of agents in the network


parser = argparse.ArgumentParser(description='Simulate repeated traffic routing game')
parser.add_argument("Algo", help="algorithm used by the agents (GPMW, cGPMW, RobustLinExp3, ...)", type = str)
parser.add_argument("--version", help="version of the algorithm (only for cGPMW)", type = int, default= 0)
parser.add_argument("--reoptimize_kernels", help="reoptimize or not kernel hyperparams", type = bool, default = False)
parser.add_argument("--num_controlled", help="number of learning agents", type = int, default= N)
parser.add_argument("--runs", help="number of runs", type = int, default= 1)
args = parser.parse_args()

" Parameters "
T = 150                       # number of iterations
Algo = args.Algo              # Choose between GPMW, cGPMW, or RobustLinExp3
num_controlled_players = args.num_controlled  #Set this to 0 to force each agent to choose the free-flow route (No-Learning)
C = 10 # Number of different contexts
# Only for cGPMW or GPMW:
version = args.version
#version = 0      # rule (5) in the paper
#version = 1      # treat contexts independently
#version = 2      # Use Lipschitzness ( rule (4) in the paper )
poly_degree = 4   # degree of polynomial kernel used by GPMW and cGPMW
reoptimize_kernels = args.reoptimize_kernels


" #################################################  START SIMULATION #######################################################  "

print('ALGORITHM: '+ str(Algo) + ' version ' + str(version))
Runs = args.runs
idxs_controlled =  random.sample( range(0, N),  num_controlled_players)

Losses_runs = []
Total_occupancies_runs = []
additional_congestions_runs = []
for run in np.arange(0,Runs):
    np.random.seed(17)
    random.seed(17)

    " Sample random Network capacities "
    original_capacities = np.array(SiouxNetwork_data.Capacities)
    Capacities = []
    for c in range(C):
        perturbed_capacities = np.multiply( 0.7*np.ones(original_capacities.shape) + 0.5 *np.random.random_sample(original_capacities.shape), np.array(original_capacities))
        Capacities.append(np.array(perturbed_capacities))
    np.random.seed(run)
    random.seed(run)
    Contexts = np.random.randint(0,C, (1,T)).squeeze()

    " Estimate maximum and minimum traveltimes (to scale rewards in [0,1])"
    M = 100
    max_traveltimes = np.zeros(N)
    min_traveltimes = 1e8 * np.ones(N)
    Capacities_rand = []
    Outcomes = []
    Payoffs = []
    for i in range(M):
        outcome = np.zeros(N)  # all play first action by default
        for p in idxs_controlled:
            outcome[p] = np.random.randint(len(Strategy_vectors[p]))
        capacities =  np.array(Capacities[np.random.randint(0, C)])
        traveltimes = Compute_traveltimes(SiouxNetwork_data, Strategy_vectors, outcome.astype(int), 'all', capacities)
        max_traveltimes = np.maximum(max_traveltimes, traveltimes + 0.01)
        min_traveltimes = np.minimum(min_traveltimes, traveltimes - 0.01)
        Capacities_rand.append(capacities)
        Outcomes.append(outcome)
        Payoffs.append(-traveltimes)

    sigmas = 0.001 * (max_traveltimes-min_traveltimes)
    Kernels = [None] * N
    if Algo == 'GPMW' or Algo == 'cGPMW' or Algo == 'cGPMWpar':
        Kernels, list_of_param_arrays = Optimize_Kernels(reoptimize_kernels, Algo, Kernels, idxs_controlled, Strategy_vectors, sigmas,
                                                         poly_degree, Outcomes, Capacities_rand, Payoffs)

    "  Compute contexts' covariance matrix (serves as input for RobustLinExp3 [Neu et al. 2020])"
    Z = []
    for c in range(C):
        Z.append(np.array(Capacities[c]))
    z_mean = np.mean(Z, axis=0)
    Sigma = np.zeros((len(Z[0]), len(Z[0])))
    for c in range(C):
        Sigma = Sigma + 1 / C * np.outer(Z[c] - z_mean, Z[c] - z_mean)

    " Initialize Players "
    Players = Initialize_Players(N, OD_pairs, Strategy_vectors, min_traveltimes, max_traveltimes, idxs_controlled, T, Algo, version, Sigma, Kernels, sigmas, C, Capacities)

    " Simulate Game "
    np.random.seed(run)
    random.seed(run)
    Game_data, Total_occupancies, addit_Congestions = Simulate_Game(run, Players, T, SiouxNetwork_data, Strategy_vectors , sigmas , Capacities, Contexts)

    ############## Save data over multiple runs ##################
    """ Save Results  """
    if num_controlled_players == 0:
        file = open('Stored_computations/stored_data_no_learning_run' + str(run)+'.pckl', 'wb')
    else:
        file = open('Stored_computations/stored_data_' + str(Algo) +'_version'+ str(version) + '_run' + str(run)+ '.pckl', 'wb')
    pickle.dump([Game_data.Incurred_losses, addit_Congestions, Total_occupancies, idxs_controlled], file)
    file.close()

    additional_congestions_runs.append(np.squeeze(addit_Congestions))
    Total_occupancies_runs.append(Total_occupancies)
    Losses = np.vstack(Game_data.Incurred_losses)
    Losses_runs.append(Losses)
    if Runs-run > 1:
        del(Game_data)
        del(Players)

"""  Average over Runs  """
mean_losses = np.mean(Losses_runs, axis = 0)
avg_mean_losses = np.mean(mean_losses[:,:], axis = 1)

mean_additional_congestions = np.mean(additional_congestions_runs, axis = 0)
avg_mean_additional_congestions = np.mean(mean_additional_congestions, axis = 1)

Actions_played = np.vstack(Game_data.Played_actions)

print('Avg mean loss: ' + str(np.mean(avg_mean_losses)))
print('Avg mean congestion: ' + str(np.mean(avg_mean_additional_congestions)))

if version == 2:
    num_balls = []
    for i in idxs_controlled:
        if Players[i].type == 'cHedge' or Players[i].type == 'cGPMW' or Players[i].type == 'cGPMWpar':
            num_balls.append(len(Players[i].contexts))
    print(num_balls)
