# Contextual Games: Multi-Agent Learning with Side Information


This repository contains the code associated to the paper:
> **Contextual Games: Multi-Agent Learning with Side Information**
> *Pier Giuseppe Sessa, Ilija Bogunovic, Andreas Krause, Maryam Kamgarpour*.
> Neural Information Processing Systems (NeurIPS), 2020.

Usage
-- 

You can install the required dependences via: 
```setup
pip install -r requirements.txt
```

The script `repeated_routing.py` simulates the repeated routing game between the agents in the network and can be run as: 
```setup
python repeated_routing.py ALGO --runs N_RUNS
```
where `ALGO` is the algorithm used by the agents (see `algorithms.py` for possible baselines). The overall simulation is repeated `N_RUNS` times.
You can also use the bash script `running_bash` to run multiple simulations.
Resulting averaged losses and congestion can be plotted using the script `plotting.py`.
