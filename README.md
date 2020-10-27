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

The script `Repeated_Routing.py` simulates the repeated traffic routing game between the agents in the network. The routing algorithm to be used by the agents should be manually specified as a parameter (Line 38 in the script). 
Resulting time-averaged losses and congestion can be plotted running `Plotting.py`.
