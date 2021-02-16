import numpy as np
import matplotlib.pyplot as plt
import pickle

plt.rcParams.update({'font.size': 13})


Runs  = 15
Algolist = [ 'no_learning', 'GPMW_version0', 'RobustLinExp3_version0', 'cGPMWpar_version2', 'cGPMWpar_version0']
Colors = ['purple', 'deepskyblue', 'orange', 'green', 'black', 'pink','purple']
Markers = ['.', '*', 's','d','o','v','*']
Styles = ['-', '-', '-', '-','-', '-']

Legend = ['No-Learning', 'GP-MW', 'RobustLinExp3', 'c.GP-MW, rule (4)', 'c.GP-MW, rule (5)']
#Legend = Algolist
plt.rc('font', size = 14)


" Create Legend as separate figure "
import pylab
fig = pylab.figure()
figlegend = pylab.figure(figsize=(12,0.5))
ax = fig.add_subplot(111)
lines= []
for Algo in Algolist:
    lines += ax.plot(range(10), pylab.randn(10), color = Colors[Algolist.index(Algo)], linestyle = Styles[Algolist.index(Algo)], marker = Markers[Algolist.index(Algo)])
figlegend.legend(lines, Legend, 'center', ncol=len(Legend))
figlegend.show()
figlegend.tight_layout()
figlegend.savefig('Saved_figs/legend.png', dpi = 500)



maximum_traveltimes = 0*np.ones(528)
minimum_traveltimes = 1e3*np.ones(528)
for Algo in Algolist:
    for run in np.arange(0,Runs+0):
        f =  open('Stored_computations/stored_data_' + str(Algo) +'_run' + str(run)+ '.pckl', 'rb')
        loader = pickle.load(f)

        losses = np.vstack(loader[0])
        min_observed = np.min(np.array(losses), axis=0)
        max_observed = np.max(np.array(losses), axis=0)
        minimum_traveltimes = np.min(np.minimum(minimum_traveltimes, min_observed-0.001))
        maximum_traveltimes = np.max(np.maximum(maximum_traveltimes, max_observed+0.001))
        f.close()

" Plot Time-Averaged Losses"
fig = plt.figure(figsize=[6,4])
for Algo in Algolist:
    Time_avg_losses_runs = []
    for run in range(Runs):
        with open('Stored_computations/stored_data_' + str(Algo) +'_run' + str(run)+ '.pckl', 'rb') as f:
            loader = pickle.load(f)

            losses = np.vstack(loader[0])
            T = len(losses)
            scaled_losses = np.divide(losses - np.tile(minimum_traveltimes, (T, 1)), np.tile(maximum_traveltimes - minimum_traveltimes, (T, 1)))
            Time_avg_losses_runs.append(scaled_losses)

    mean_runs = np.mean(Time_avg_losses_runs, axis=0)
    std_runs = np.std(Time_avg_losses_runs, axis=0)
    if Algo == 'no_learning':
        idx_controlled = range(528)
    else:
        idx_controlled = loader[3]

    avg_mean_runs = np.mean(mean_runs[:, :], axis=1)
    #avg_mean_runs = np.divide(np.cumsum(avg_mean_runs), np.arange(1, len(avg_mean_runs)+1))

    plot_mean = avg_mean_runs
    T = len(plot_mean)
    plt.plot(plot_mean,  color = Colors[Algolist.index(Algo)], linestyle = Styles[Algolist.index(Algo)], marker = Markers[Algolist.index(Algo)], markevery = 10)
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_facecolor("gainsboro")
    plt.grid(True, which="both", ls="-", color = 'white')
#plt.legend(Legend, loc = 'best')
plt.xlabel('rounds')
plt.title('average losses')
fig.show()
fig.tight_layout()
fig.savefig('Saved_figs/avg_losses.png', dpi = 500)


" Plot Time-Averaged Congestion"

fig = plt.figure(figsize=[6,4])
for Algo in Algolist:
    Congestions_runs = []
    for run in range(Runs):
        with open('Stored_computations/stored_data_' + str(Algo) +'_run' + str(run)+ '.pckl', 'rb') as f:
            loader = pickle.load(f)
            Congestions_runs.append(loader[1])

    mean_congestions = np.mean(Congestions_runs, axis=0)
    std_congestions = np.std(Congestions_runs, axis=0)
    avg_mean_congestions = np.mean(mean_congestions[:, :], axis=1)
    #avg_mean_congestions = np.divide(np.cumsum(avg_mean_congestions), np.arange(1, len(avg_mean_congestions)+1))

    plot_mean = avg_mean_congestions
    T = len(plot_mean)
    plt.plot(plot_mean,  color = Colors[Algolist.index(Algo)], linestyle = Styles[Algolist.index(Algo)], marker = Markers[Algolist.index(Algo)], markevery = 10)
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_facecolor("gainsboro")
    plt.grid(True, which="both", ls="-", color='white')
#plt.legend(Legend,  loc = 'upper right')
plt.xlabel('rounds')
plt.title('average network congestion')
fig.show()
fig.tight_layout()
fig.savefig('Saved_figs/avg_congestions.png', dpi = 500)


