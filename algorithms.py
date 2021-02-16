import GPy
from network_fcns import* # Load the needed Sioux Falls Network parameters and functions


# Algorithms: GPMW [Sessa et al. 2019],
#             RobustLinExp3 [Neu et al. 2020] for **linear** contextual bandits,
#             cGPMW [Sessa et al. 2020] and cGPMWpar (a parallel (recursive) implementation of cGPMW which knows the set of contexts in advance)
#             Hedge [Freund and Schapire 1997] and cHedge are the full-information equivalent to GPMW and cGPMW
#             EXP3P [Auer et al. 2002] is a bandit non-contextual baseline

class Player_GPMW:
    def __init__(self, K, T, min_payoff, max_payoff, my_strategy_vecs, kernel, sigma_e):
        self.type = "GPMW"
        self.K = K
        self.min_payoff = min_payoff
        self.max_payoff = max_payoff
        self.weights = np.ones(K)
        self.T = T
        self.idx_nonzeros = np.where(np.sum(my_strategy_vecs, axis=0) != 0)[0]

        self.cum_losses= np.zeros(K)
        self.mean_rewards_est = np.zeros(K)
        self.std_rewards_est = np.zeros(K)
        self.ucb_rewards_est = np.zeros(K)
        self.gamma_t = np.sqrt(8*np.log(K) / T)
        self.kernel = kernel
        self.sigma_e = sigma_e
        self.strategy_vecs = my_strategy_vecs

        self.history_payoffs = np.empty((0, 1))
        self.history = np.empty((0, len(self.idx_nonzeros) * 2))
        self.demand = np.max(my_strategy_vecs[0])

    def mixed_strategy(self):
        return self.weights / np.sum(self.weights)

    def sample_action(self):
        return np.random.choice(self.K, p=np.array(self.mixed_strategy()))

    def Update(self, played_action, total_occupancies, payoff, Capacities_t):

        self.history_payoffs = np.vstack((self.history_payoffs, payoff))
        self.history = np.vstack((self.history, np.concatenate((self.strategy_vecs[played_action][self.idx_nonzeros].T, total_occupancies[self.idx_nonzeros].T), axis=1)))

        beta_t = 0.5

        m = GPy.models.GPRegression(self.history, self.history_payoffs, self.kernel)
        m.Gaussian_noise.fix(self.sigma_e ** 2)

        other_occupancies = total_occupancies[self.idx_nonzeros] - self.strategy_vecs[played_action][self.idx_nonzeros]
        for a1 in range(self.K):
            x1 = self.strategy_vecs[a1][self.idx_nonzeros]
            x2 = other_occupancies + x1
            mu, var = m.predict(np.concatenate((x1.T, x2.T), axis=1))
            sigma = np.sqrt(np.maximum(var, 1e-6))

            self.ucb_rewards_est[a1] = mu + beta_t * sigma
            self.mean_rewards_est[a1] = mu
            self.std_rewards_est[a1] = sigma

        payoffs = np.array(self.ucb_rewards_est)
        payoffs = np.maximum(payoffs, self.min_payoff * np.ones(self.K))
        payoffs = np.minimum(payoffs, self.max_payoff * np.ones(self.K))
        payoffs_scaled = np.array((payoffs - self.min_payoff) / (self.max_payoff - self.min_payoff))
        losses = np.ones(self.K) - np.array(payoffs_scaled)
        self.cum_losses = self.cum_losses + losses

        gamma_t = self.gamma_t
        self.weights = np.exp(np.multiply(gamma_t, -self.cum_losses))

    


class Player_RobustLinExp3:
    def __init__(self, K, T, min_payoff, max_payoff, Capacities, numberofcontexts, my_strategy_vecs, Sigma, version=None):
        self.type = "RobustLinExp3"
        self.K = K
        self.min_payoff = min_payoff
        self.max_payoff = max_payoff
        self.idx_nonzeros = np.where(np.sum(my_strategy_vecs, axis=0) != 0)[0]
        self.weights = 1/self.K * np.ones((T,K)) #in the worst case there are T different contexts
        self.T = T
        self.gamma_t = 0.2 #T**(-2/3)*(K*len(self.idx_nonzeros))**(-1/3)*(np.log(K))**(2/3)
        self.eta =   0.6#T**(-1/3)*(K*len(self.idx_nonzeros)*np.log(K))**(1/2)  #mixing factor
        self.C = numberofcontexts
        self.contexts = []
        self.c_idx_t = 0
        self.Sigma = Sigma[np.ix_(self.idx_nonzeros, self.idx_nonzeros)] # Covariance matrix of contexts' distribution
        self.Capacities = Capacities

    def mixed_strategy(self,context_idx):
        return self.weights[context_idx,:] / np.sum(self.weights[context_idx,:])

    def sample_action(self,context_idx, Capacities_t):
        self.c_idx_t = context_idx

        weights = np.array(self.mixed_strategy( self.c_idx_t))
        weights = (1-self.eta)*weights + self.eta/self.K
        return np.random.choice(self.K, p=np.array(weights))

    def Update(self, played_actions, context_idx, player_idx, SiouxNetwork_data_original, Strategy_vectors, original_capacities, Capacities_t, traveltime_t):
        contexts_range = range(self.C)  # update strategy for all contexts

        a_t = played_actions[player_idx]
        z_t = Capacities_t[self.idx_nonzeros] #context at round t
        loss_t = traveltime_t # loss at time t
        weights = np.array(self.mixed_strategy(context_idx))
        prob = (1 - self.eta) * weights[a_t] + self.eta / self.K  #prob of playing action a_t at time t
        for c in contexts_range:  #update in parallel the weights for all the contexts
            Capacities = np.array(self.Capacities[c])
            losses_hindsight = np.zeros(self.K)
            z_c = Capacities[self.idx_nonzeros]
            losses_hindsight[a_t] = np.dot(z_c.T , (1/prob) * np.dot(self.Sigma**(-1), z_t) * loss_t)

            payoffs = -losses_hindsight  
            payoffs = np.maximum(payoffs, self.min_payoff * np.ones(self.K))
            payoffs = np.minimum(payoffs, self.max_payoff * np.ones(self.K))
            payoffs_scaled = np.array((payoffs - self.min_payoff) / (self.max_payoff - self.min_payoff))
            losses = np.ones(self.K) - np.array(payoffs_scaled)
            self.weights[c,a_t] = np.multiply(self.weights[c,a_t], np.exp(np.multiply(self.gamma_t, -losses[a_t])))



class Player_cGPMW:
    def __init__(self, K, T, min_payoff, max_payoff, Capacities, my_strategy_vecs, kernel,sigma_e, version):
        self.type = "cGPMW"
        self.K = K
        self.min_payoff = min_payoff
        self.max_payoff = max_payoff
        self.weights = np.ones(K)
        self.T = T
        self.idx_nonzeros = np.where(np.sum(my_strategy_vecs, axis=0) != 0)[0]


        self.gamma_t = np.sqrt(8*np.log(K) / T)
        self.kernel = kernel
        self.sigma_e = sigma_e
        self.strategy_vecs = my_strategy_vecs

        self.history_payoffs = np.empty((0, 1))
        self.history_played_actions = np.empty((0, 1))
        self.history_occupancies = []
        self.history = np.empty((0, len(self.idx_nonzeros) * 2))

        self.demand = np.max(my_strategy_vecs[0])
        self.contexts = []
        self.idx_balls = []
        self.version  = version
        self.Capacities = Capacities

    def mixed_strategy(self):
        return self.weights / np.sum(self.weights)

    def sample_action(self):
        return np.random.choice(self.K, p=np.array(self.mixed_strategy()))

    def Update_history(self,played_action, payoff, occupancies , capacities):
        self.history_played_actions = np.vstack((self.history_played_actions, played_action))
        self.history_payoffs = np.vstack((self.history_payoffs, payoff))
        self.history_occupancies.append(occupancies)
        self.history = np.vstack((self.history, np.concatenate(
            (self.strategy_vecs[played_action][self.idx_nonzeros].T, np.divide(occupancies[self.idx_nonzeros], capacities[self.idx_nonzeros]).T), axis=1)))

    def Compute_strategy(self, capacities_t):

        if self.version ==2:
            epsilon  = 30*len(self.idx_nonzeros)
            context_t = np.array(capacities_t[self.idx_nonzeros])
            if len(self.contexts) == 0:
                self.contexts.append(context_t)
            distances = np.array([np.linalg.norm(context_t - self.contexts[c], 1) for c in range(len(self.contexts))])
            if distances.min() < epsilon:
                c_idx_t = distances.argmin()
            else:
                c_idx_t = len(self.contexts)
                self.contexts.append(context_t)
            self.idx_balls.append(c_idx_t)


        beta_t = 0.5

        cum_payoffs_scaled = np.zeros(self.K)
        num_t = 0

        add = 0
        # time tau = 0
        if self.version == 2 and c_idx_t == self.idx_balls[0]:
            add = 1
        if self.version == 0:
            add == 1
        if add == 1:
            num_t = num_t + 1
            payoffs = np.zeros(self.K)
            other_occupancies_0 = self.history_occupancies[0][self.idx_nonzeros] - self.strategy_vecs[np.squeeze(self.history_played_actions[0].astype(int))][ self.idx_nonzeros]
            for a1 in range(self.K):
                x1 = self.strategy_vecs[a1][self.idx_nonzeros]
                x2 = np.divide( other_occupancies_0 + x1, capacities_t[self.idx_nonzeros] )
                mu = 0
                var = self.kernel.K(np.concatenate((x1.T, x2.T), axis=1), np.concatenate((x1.T, x2.T), axis=1))
                sigma = np.sqrt(np.maximum(var, 1e-6))

                payoffs[a1] = mu + beta_t * sigma
            payoffs = np.maximum(payoffs, self.min_payoff * np.ones(self.K))
            payoffs = np.minimum(payoffs, self.max_payoff * np.ones(self.K))
            payoffs_scaled = np.array((payoffs - self.min_payoff) / (self.max_payoff - self.min_payoff))
            cum_payoffs_scaled = cum_payoffs_scaled + payoffs_scaled

        t = len(self.history_played_actions)
        if t > 0:
            m = GPy.models.GPRegression(self.history[:,:], self.history_payoffs[:], self.kernel)
            m.Gaussian_noise.fix(self.sigma_e ** 2)

            for tau in range(1,t):
                add = 0
                if self.version == 2 and c_idx_t == self.idx_balls[tau]:
                    add = 1
                if self.version == 0:
                    add = 1
                if add == 1:
                    num_t = num_t + 1
                    m.set_XY(self.history[0:tau+1,:], self.history_payoffs[0:tau+1])
                    other_occupancies = self.history_occupancies[tau][self.idx_nonzeros] - self.strategy_vecs[np.squeeze(self.history_played_actions[tau].astype(int))][self.idx_nonzeros]
                    payoffs = np.zeros(self.K)
                    for a1 in range(self.K):
                        x1 = self.strategy_vecs[a1][self.idx_nonzeros]
                        x2 = np.divide( other_occupancies + x1, capacities_t[self.idx_nonzeros] )

                        mu, var = m.predict(np.concatenate((x1.T, x2.T), axis=1))
                        sigma = np.sqrt(np.maximum(var, 1e-6))

                        payoffs[a1] = mu + beta_t * sigma
                    payoffs = np.maximum(payoffs, self.min_payoff * np.ones(self.K))
                    payoffs = np.minimum(payoffs, self.max_payoff * np.ones(self.K))
                    payoffs_scaled = np.array((payoffs - self.min_payoff) / (self.max_payoff - self.min_payoff))
                    cum_payoffs_scaled = cum_payoffs_scaled + payoffs_scaled

        # Compute strategy at time t
        if self.version ==2:
            gamma_t = 2*np.sqrt(np.log(self.K)/num_t)
        else:
            gamma_t = self.gamma_t
        cum_losses = num_t * np.ones(self.K) - np.array(cum_payoffs_scaled)
        self.weights = np.exp(np.multiply(gamma_t, -cum_losses))


class Player_cGPMWpar:
    def __init__(self, K, T, min_payoff, max_payoff, Capacities, numberofcontexts, my_strategy_vecs, version, sigma_e, Kernel):
        self.type = "cGPMWpar"
        self.K = K
        self.min_payoff = min_payoff
        self.max_payoff = max_payoff
        self.cum_payoffs_scaled = np.zeros((T, K))  # in the worst case there are T different contexts
        self.T = T
        self.gamma_t = np.sqrt(8 * np.log(K) / T)
        self.C = numberofcontexts
        self.version = version
        self.idx_nonzeros = np.where(np.sum(my_strategy_vecs, axis=0) != 0)[0]
        self.contexts = []
        self.c_idx_t = 0
        self.counts = np.zeros((T, 1))

        self.kernel = Kernel
        self.strategy_vecs = my_strategy_vecs
        self.history_payoffs = np.empty((0, 1))
        self.history_played_actions = np.empty((0, 1))
        self.history_occupancies = []
        self.history = np.empty((0, len(self.idx_nonzeros) * 2))
        self.Capacities = Capacities
        self.sigma_e = sigma_e

    def mixed_strategy(self, c_idx):
        if self.version == 2:
            gamma_t = 2 * np.sqrt(np.log(self.K) / (self.counts[c_idx]))
        else:
            gamma_t = self.gamma_t
        losses = (self.counts[c_idx] - 1) * np.ones(self.K) - np.array(self.cum_payoffs_scaled[c_idx, :])
        weights = np.exp(np.multiply(gamma_t, -losses))
        return weights / np.sum(weights)

    def sample_action(self, context_idx, Capacities_t):
        if self.version == 2:
            epsilon = 25*len(self.idx_nonzeros)
            capacities = np.array(Capacities_t[self.idx_nonzeros])
            if len(self.contexts) == 0:
                self.contexts.append(capacities)
            distances = np.array([np.linalg.norm(capacities - self.contexts[c], 1) for c in range(len(self.contexts))])
            if distances.min() < epsilon:
                self.c_idx_t = distances.argmin()
            else:
                self.c_idx_t = len(self.contexts)
                self.contexts.append(capacities)
            self.counts[self.c_idx_t] = self.counts[self.c_idx_t] + 1
        else:
            self.c_idx_t = context_idx
            for c in range(len(self.counts)):
                self.counts[c] = self.counts[c] + 1

        return np.random.choice(self.K, p=np.array(self.mixed_strategy(self.c_idx_t)))

    def Update_history(self,played_action, payoff, occupancies , capacities):
        self.history_played_actions = np.vstack((self.history_played_actions, played_action))
        self.history_payoffs = np.vstack((self.history_payoffs, payoff))
        self.history_occupancies.append(occupancies)
        self.history = np.vstack((self.history, np.concatenate(
            (self.strategy_vecs[played_action][self.idx_nonzeros].T, np.divide(occupancies[self.idx_nonzeros], capacities[self.idx_nonzeros]).T), axis=1)))


    def Update(self, played_actions, context_idx, player_idx, SiouxNetwork_data_original, original_capacities,
                   Capacities_t, Strategy_vectors):
        if self.version == 0:
            contexts_range = range(self.C)  # update strategy for all contexts
        elif self.version == 1:
            contexts_range = [context_idx]  # update strategy only for context_t
        elif self.version == 2:
            contexts_range = [self.c_idx_t]

        m = GPy.models.GPRegression(self.history, self.history_payoffs, self.kernel)
        m.Gaussian_noise.fix(self.sigma_e ** 2)

        beta_t = 0.5
        for c in contexts_range:
            Capacities = np.array(self.Capacities[c])
            if self.version == 2:
                Capacities = np.array(Capacities_t)
            other_occupancies = self.history_occupancies[-1][self.idx_nonzeros] - self.strategy_vecs[np.squeeze(self.history_played_actions[-1].astype(int))][self.idx_nonzeros]
            payoffs = np.zeros(self.K)
            payoffs_lcb = np.zeros(self.K)
            for a1 in range(self.K):
                x1 = self.strategy_vecs[a1][self.idx_nonzeros]
                x2 = np.divide(other_occupancies + x1, Capacities[self.idx_nonzeros])
                mu, var = m.predict(np.concatenate((x1.T, x2.T), axis=1))
                sigma = np.sqrt(np.maximum(var, 1e-6))
                payoffs[a1] = mu + beta_t * sigma
                payoffs_lcb[a1] = mu - beta_t * sigma

            payoffs = np.maximum(payoffs, self.min_payoff * np.ones(self.K))
            payoffs = np.minimum(payoffs, self.max_payoff * np.ones(self.K))
            payoffs_scaled = np.array((payoffs - self.min_payoff) / (self.max_payoff - self.min_payoff))
            self.cum_payoffs_scaled[c, :] = self.cum_payoffs_scaled[c, :] + payoffs_scaled


class Player_EXP3P:
    def __init__(self, K, T, min_payoff, max_payoff):
        self.type = "EXP3P"
        self.K = K
        self.P = K  # num of policies = num of actions
        self.min_payoff = min_payoff
        self.max_payoff = max_payoff
        self.T = T
        self.weights = np.ones(K)
        self.rewards_est = np.zeros(K)

        self.beta = np.sqrt(np.log(self.K) / (self.T * self.K))
        self.gamma = 1.05 * np.sqrt(np.log(self.K) * self.K / self.T)
        self.eta = 0.95 * np.sqrt(np.log(self.K) / (self.T * self.K))
        assert self.K == 1 or (self.beta > 0 and self.beta < 1 and self.gamma > 0 and self.gamma < 1)

    def mixed_strategy(self):
        return self.weights / np.sum(self.weights)

    def sample_action(self):
        return np.random.choice(self.K, p=np.array(self.mixed_strategy()))

    def Update(self, played_a, payoff):
        prob = self.weights[played_a] / np.sum(self.weights)
        # assert  payoff > self.min_payoff - 1e-2 and payoff < self.max_payoff +1e-2, "min payoff = "+ str(payoff) + " lb = " + str(self.min_payoff)
        payoff = np.maximum(payoff, self.min_payoff)
        payoff = np.minimum(payoff, self.max_payoff)
        payoff_scaled = np.array((payoff - self.min_payoff) / (self.max_payoff - self.min_payoff))

        self.rewards_est = self.rewards_est + self.beta * np.divide(np.ones(self.K),
                                                                    self.weights / np.sum(self.weights))
        self.rewards_est[played_a] = self.rewards_est[played_a] + payoff_scaled / prob

        self.weights = np.exp(np.multiply(self.eta, self.rewards_est))
        self.weights = self.weights / np.sum(self.weights)
        self.weights = (1 - self.gamma) * self.weights + self.gamma / self.K * np.ones(self.K)



class Player_Hedge:
    def __init__(self, K, T, min_payoff, max_payoff):
        self.type = "Hedge"
        self.K = K
        self.min_payoff = min_payoff
        self.max_payoff = max_payoff
        self.weights = np.ones(K)
        self.T = T
        self.gamma_t = np.sqrt(8 * np.log(K) / T)

    def mixed_strategy(self):
        return self.weights / np.sum(self.weights)

    def sample_action(self):
        return np.random.choice(self.K, p=np.array(self.mixed_strategy()))

    def Update(self, played_actions, player_idx, SiouxNetwork_data_original, Capacities_t, Strategy_vectors):
        losses_hindsight = np.zeros(self.K)
        for a in range(self.K):
            modified_outcome = np.array(played_actions)
            modified_outcome[player_idx] = a
            losses_hindsight[a] = Compute_traveltimes(SiouxNetwork_data_original, Strategy_vectors, modified_outcome, player_idx,Capacities_t)

        payoffs  = -losses_hindsight 
        payoffs = np.maximum(payoffs, self.min_payoff * np.ones(self.K))
        payoffs = np.minimum(payoffs, self.max_payoff * np.ones(self.K))
        payoffs_scaled = np.array((payoffs - self.min_payoff) / (self.max_payoff - self.min_payoff))
        losses = np.ones(self.K) - np.array(payoffs_scaled)
        self.weights = np.multiply(self.weights, np.exp(np.multiply(self.gamma_t, -losses)))
        self.weights = self.weights / np.sum(self.weights)  # To avoid numerical errors when the weights become too small


class Player_cHedge:
    def __init__(self, K, T, min_payoff, max_payoff, Capacities, numberofcontexts, my_strategy_vecs, version):
        self.type = "cHedge"
        self.K = K
        self.min_payoff = min_payoff
        self.max_payoff = max_payoff
        self.cum_payoffs_scaled = np.zeros((T,K)) #in the worst case there are T different contexts
        self.T = T
        self.gamma_t = np.sqrt(8 * np.log(K) / T)
        self.C = numberofcontexts
        self.version  = version
        self.idx_nonzeros = np.where(np.sum(my_strategy_vecs, axis=0) != 0)[0]
        self.contexts = []
        self.c_idx_t = 0
        self.counts = np.zeros((T,1))
        self.Capacities = Capacities

    def mixed_strategy(self,context_idx):
        if self.version ==2:
            gamma_t = 2*np.sqrt(np.log(self.K)/self.counts[context_idx])
        else:
            gamma_t = self.gamma_t
        losses = (self.counts[context_idx] - 1) * np.ones(self.K) - np.array(self.cum_payoffs_scaled[context_idx, :])
        weights = np.exp(np.multiply(gamma_t, -losses))
        return weights / np.sum(weights)

    def sample_action(self,context_idx, Capacities_t):
        if self.version  == 2:
            epsilon = 40*len(self.idx_nonzeros)
            capacities = np.array(Capacities_t[self.idx_nonzeros])
            if len(self.contexts)==0:
                self.contexts.append(capacities)
            distances =  np.array([np.linalg.norm(capacities - self.contexts[c], 1) for c in range(len(self.contexts)) ] )
            if distances.min() < epsilon:
                self.c_idx_t = distances.argmin()
            else:
                self.c_idx_t = len(self.contexts)
                self.contexts.append(capacities)
            self.counts[self.c_idx_t] = self.counts[self.c_idx_t]+1
        else:
            self.c_idx_t = context_idx
            for c in range(len(self.counts)):
                self.counts[c] = self.counts[c]+1

        return np.random.choice(self.K, p=np.array(self.mixed_strategy( self.c_idx_t)))

    def Update(self, played_actions, context_idx, player_idx, SiouxNetwork_data_original,original_capacities, Capacities_t, Strategy_vectors):
        if self.version == 0:
            contexts_range = range(self.C)  # update strategy for all contexts
        elif self.version == 1:
            contexts_range = [context_idx]  # update strategy only for context_t
        elif self.version == 2:
            contexts_range = [self.c_idx_t]

        for c in contexts_range:
            Capacities = np.array(self.Capacities[c])
            if self.version == 2:
                Capacities = np.array(Capacities_t)
            losses_hindsight = np.zeros(self.K)
            for a in range(self.K):
                modified_outcome = np.array(played_actions)
                modified_outcome[player_idx] = a
                losses_hindsight[a] = Compute_traveltimes(SiouxNetwork_data_original, Strategy_vectors, modified_outcome, player_idx, Capacities)

            payoffs = -losses_hindsight  
            payoffs = np.maximum(payoffs, self.min_payoff * np.ones(self.K))
            payoffs = np.minimum(payoffs, self.max_payoff * np.ones(self.K))
            payoffs_scaled = np.array((payoffs - self.min_payoff) / (self.max_payoff - self.min_payoff))
            self.cum_payoffs_scaled[c,:] = self.cum_payoffs_scaled[c,:] + payoffs_scaled
            
