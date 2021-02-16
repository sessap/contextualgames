from algorithms import *


class GameData:
     def __init__(self,N):
        self.Played_actions   = []
        self.Mixed_strategies = []  
        self.Incurred_losses   = []  
        self.Regrets          =  []  
        self.Cum_losses        = [()]*N

def Initialize_Players(N, OD_pairs, Strategy_vectors, min_traveltimes,  max_traveltimes, idxs_controlled, T, Algo, version, Sigma = None,  Kernels = None,  sigmas = None, numberofcontexts = None, Capacities= None):
    Players = []
    for i in range(N):
        K_i = len(Strategy_vectors[i])
        min_payoff = - max_traveltimes[i]
        max_payoff = - min_traveltimes[i]
        if i in idxs_controlled and K_i > 1:
            if Algo == 'Hedge':
                Players.append(Player_Hedge(K_i, T, min_payoff, max_payoff))
            elif Algo == 'cHedge':
                Players.append(Player_cHedge(K_i, T, min_payoff, max_payoff, Capacities, numberofcontexts, Strategy_vectors[i], version))
            elif Algo == 'cGPMWpar':
                Players.append(Player_cGPMWpar(K_i, T, min_payoff, max_payoff, Capacities, numberofcontexts, Strategy_vectors[i], version, sigmas[i],Kernels[i]))
            elif Algo == 'EXP3P':
                Players.append(Player_EXP3P(K_i, T, min_payoff, max_payoff))
            elif Algo == 'RobustLinExp3':
                Players.append(Player_RobustLinExp3(K_i, T, min_payoff, max_payoff, Capacities, numberofcontexts, Strategy_vectors[i], Sigma, version))
            elif Algo == 'GPMW':
                Players.append(Player_GPMW(K_i, T, min_payoff, max_payoff, Strategy_vectors[i], Kernels[i], sigmas[i]))
            elif Algo == 'cGPMW':
                Players.append(Player_cGPMW(K_i, T, min_payoff, max_payoff, Capacities, Strategy_vectors[i], Kernels[i], sigmas[i], version))
        else:
            K_i = 1
            Players.append(Player_Hedge(K_i, T, min_payoff, max_payoff))

        Players[i].OD_pair = OD_pairs[i]

    return Players


def Simulate_Game(run, Players, T, SiouxNetwork_data_original, Strategy_vectors , sigmas , Capacities, Contexts = None):
    N = len(Players)
    Game_data = GameData(N)
    for i in range(N):
        Game_data.Cum_losses[i] = np.zeros(Players[i].K)

    Total_occupancies = []
    addit_Congestions = []
    original_capacities = np.array(SiouxNetwork_data_original.Capacities)
    for t in range(T):
        Capacities_t = np.array(Capacities[Contexts[t]])

        " Compute played actions "
        played_actions_t = np.empty(N, 'i')
        for i in range(N):
            if Players[i].type == 'cHedge' or Players[i].type == 'cGPMWpar' or Players[i].type == 'RobustLinExp3':
                played_actions_t[i] = Players[i].sample_action(Contexts[t], Capacities_t)
            else:
                if Players[i].type == 'cGPMW' and t > 0:
                    Players[i].Compute_strategy(Capacities_t)
                played_actions_t[i] = Players[i].sample_action()
        Game_data.Played_actions.append(played_actions_t)

        " Assign payoffs "
        losses_t = Compute_traveltimes(SiouxNetwork_data_original, Strategy_vectors, Game_data.Played_actions[t], 'all', Capacities_t)
        Game_data.Incurred_losses.append(losses_t)

        Total_occupancies.append(np.sum([Strategy_vectors[i][Game_data.Played_actions[t][i]] for i in range(N)], axis=0))

        congestions = 0.15 * np.power(np.divide(Total_occupancies[t], Capacities_t), 4)
        addit_Congestions.append(congestions)

        " Update players next mixed strategy "
        for i in range(N):
            if Players[i].type == "EXP3P":
                noisy_loss = Game_data.Incurred_losses[t][i] + np.random.normal(0, sigmas[i], 1)
                Players[i].Update(Game_data.Played_actions[t][i], -noisy_loss)

            if Players[i].type == "Hedge":
                Players[i].Update( Game_data.Played_actions[t], i , SiouxNetwork_data_original, Capacities_t,  Strategy_vectors)

            if Players[i].type == "cHedge":
                Players[i].Update( Game_data.Played_actions[t], Contexts[t], i , SiouxNetwork_data_original, original_capacities, Capacities_t,  Strategy_vectors)

            if Players[i].type == "RobustLinExp3":
                noisy_loss = Game_data.Incurred_losses[t][i] + np.random.normal(0, sigmas[i], 1)
                Players[i].Update( Game_data.Played_actions[t], Contexts[t], i , SiouxNetwork_data_original, Strategy_vectors, original_capacities, Capacities_t, noisy_loss)

            if Players[i].type == "GPMW":
                noisy_loss = Game_data.Incurred_losses[t][i] + np.random.normal(0, sigmas[i], 1)
                Players[i].Update(Game_data.Played_actions[t][i], Total_occupancies[-1], -noisy_loss, Capacities_t)

            if Players[i].type == "cGPMW":
                noisy_loss = Game_data.Incurred_losses[t][i] + np.random.normal(0, sigmas[i], 1)
                Players[i].Update_history(Game_data.Played_actions[t][i], -noisy_loss, Total_occupancies[-1], Capacities_t )

            if Players[i].type == "cGPMWpar":
                noisy_loss = Game_data.Incurred_losses[t][i] + np.random.normal(0, sigmas[i], 1)
                Players[i].Update_history(Game_data.Played_actions[t][i], -noisy_loss, Total_occupancies[-1], Capacities_t )
                Players[i].Update( Game_data.Played_actions[t], Contexts[t], i , SiouxNetwork_data_original, original_capacities, Capacities_t,  Strategy_vectors)

        avg_cong = np.sum(np.mean(addit_Congestions, axis = 1))/len(addit_Congestions)
        print(Players[2].type + ' run: ' + str(run+1) + ', time: '+ str(t) + ', Avg cong. %.2f' % avg_cong, end= '\r')

    return Game_data, Total_occupancies , addit_Congestions


def Optimize_Kernels(reoptimize , Algo, Kernels, idxs_controlled, Strategy_vectors, sigmas, poly_degree,  Outcomes, Capacities, Payoffs):
    N = len(Kernels)
    list_of_param_arrays = np.load('list_of_param_arrays_' + Algo +'.npy')
    for p in idxs_controlled:
        if Kernels[p] == None:
            idx_nonzeros = np.where(np.sum(Strategy_vectors[p], axis=0) != 0)[0]
            dim = len(idx_nonzeros)
            if reoptimize == False:
                loaded_params = list_of_param_arrays[p]
                kernel_1 = GPy.kern.Poly(input_dim=dim, variance= loaded_params[0], scale=loaded_params[1], bias=loaded_params[2], order=1.,
                                         active_dims=np.arange(0, dim))
                kernel_2 = GPy.kern.Poly(input_dim=dim, variance=loaded_params[3], scale=loaded_params[4], bias=loaded_params[5], order=poly_degree,
                                         active_dims=np.arange(dim, 2 * dim))
                Kernels[p] = kernel_1 * kernel_2

            if reoptimize == True:  #Re-Optimize Kernel Hyperparameters
                kernel_1 = GPy.kern.Poly(input_dim=dim,  order=1., active_dims=np.arange(0, dim), bias = 1e-6, variance=1)
                kernel_2 = GPy.kern.Poly(input_dim=dim, order=poly_degree, active_dims=np.arange(dim, 2 * dim))
                Kernels[p] = kernel_1 * kernel_2

                if len(Strategy_vectors[p]) > 1: #Re-Optimize Kernel Hyperparameters
                    X = np.empty((0, dim * 2))
                    y = np.empty((0, 1))
                    y_true = np.empty((0, 1))
                    for a in range(500):
                        x1 = Strategy_vectors[p][int(Outcomes[a][p])]
                        occupancies = np.sum([Strategy_vectors[i][int(Outcomes[a][i])] for i in range(N)], axis=0)
                        if Algo == 'GPMW':
                            x2 = occupancies
                        else:
                            x2 = np.divide(occupancies, Capacities[a])

                        X = np.vstack((X, np.concatenate((x1[idx_nonzeros].T, x2[idx_nonzeros].T), axis=1)))
                        y = np.vstack((y, Payoffs[a][p] + 1*np.random.normal(0, sigmas[p], 1)))
                        y_true = np.vstack((y_true, Payoffs[a][p]))
                    # Fit to data using Maximum Likelihood Estimation of the parameters
                    m = GPy.models.GPRegression(X[:450, :], y[:450], Kernels[p])
                    m.Gaussian_noise.fix(sigmas[p] ** 2)
                    m.kern.poly.bias.constrain_fixed()
                    m.kern.poly.variance.constrain_fixed()
                    m.constrain_bounded(1e-6, 1e6)
                    m.optimize_restarts(num_restarts=1, max_f_eval=100)

                    if 0:
                        mu,var = m.predict(X[450:,:])
                        sigma = np.sqrt(np.maximum(var, 1e-6))
                        plt.plot(-y_true[450:])
                        plt.plot(-mu)
                        plt.plot(-(mu+sigma))
                        ax = plt.gca()
                        ax.set_yscale('log')
                        #ax.set_ylim([-1000, 0])
                        plt.show()

                        plt.plot(np.divide(abs(y_true[450:,:]-mu),abs(y_true[450:,:])))
                        plt.show()

    if reoptimize == True: # override existing parameters
        list_of_param_arrays = []
        for i in range(len(Kernels)):
            list_of_param_arrays.append(Kernels[i].param_array)
        np.save('list_of_param_arrays_' + Algo, list_of_param_arrays)

    return Kernels , list_of_param_arrays
        
        