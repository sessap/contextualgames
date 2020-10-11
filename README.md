# Contextual Games: Multiagent Learning with Side Information


This repository contains the code associated to the paper:
> **Contextual Games: Multiagent Learning with Side Information**
> *Pier Giuseppe Sessa, Ilija Bogunovic, Andreas Krause, Maryam Kamgarpour*.
> Neural Information Processing Systems (NeurIPS), 2020.

Usage
-- 

You can install the required dependences via: 
```setup
pip install -r requirements.txt
```

The script `ContextualRouting\Repeated_Routing.py` reproduces the results of the paper, simulating the repeated game between the agents in the network. The chosen routing algorithm should be manually specified as a parameter (Line 38 in the script). 
Time-averaged losses and congestion can be plotted running `ContextualRouting\Plotting.py`.
