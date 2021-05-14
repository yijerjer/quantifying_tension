# Constructing a Non-Linear Cosmological Tension Coordinate with Neural Networks

## Directory
```
.
+-- final_report/           # contains tex files of final report 
+-- notebooks/              # contains all the jupyter notebooks 
                              (unlikely to run out of the box)
|   +-- *.ipynb
+-- plots/                  # contains all the plots in the final report
|   +-- pair/               # trained PyTorch model state_dicts for 
                              parameter pair neural network
|   +-- six_2/              # trained Pytorch model state_dicts for six 
                              parameter neural network
|   +-- *.png
|   +-- toy_*.pt            # trained Pytorch model state_dicts for toy 
                              examples
+-- runs_default/           # nested sampling chains from 
                              [https://zenodo.org/record/4116393#.YJ5BIl3TUVk]
+-- getting_started.py      # Starter script to train six parameter neural 
                              network
+-- plot_*.py               # scripts to produce *.png in plots/
+-- np_utils.py             # util methods to create data points
+-- tension_net.py          # Pytorch neural network classes
+-- tension_quantify.py     # tension quantification classes, including
                              KDE, Bayes factor, Suspiciousness
+-- torch_utils.py          # class to train model and make relevant plots

```

## Getting Started
Use the `getting_started.py` script to get going with traning a neural network for six parameters