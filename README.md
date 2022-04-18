# ANN learning & Boolean complexity

This project compares ANN learning effort and logical complexity in the Boolean domain.

The repo contains the following:
- analysis: A folder with the analysis code and the plots. 
The analyses compare for each category Boolean logical complexity and the ANN learning effort.
Note that the code contains more and more complex analyses than are described in the paper.
As it turns out, the main insights are apparent with quite simple correlations.
- data: A folder meant to contain the data. It is empty because of repo space limitations, but see below for data.
- neuralNetsLearning: A folder containing the code used to train the ANNs.

All the server-side code is written for a SLURM scheduler.
Nonetheless, it gives the code needed to reproduce all experiments.

Note that in order to follow the DRY principle, this repo does not include the code used to calculate the minimal formulas. 
Instead, the code to do this can be found at [this link](https://github.com/thelogicalgrammar/inferringLoT/tree/main/booleanMinimization) in a different repo.

# Data

All the data needed to reproduce the results is in [this OSF repository](https://osf.io/gfsdq/).
Some of the other files are too large to be uploaded in OSF,
(e.g. full ANN learning data is 16GBs) 
but please ask if you'd like to have them.

# Authors

- Fausto Carcassi (First author)
- Jakub Szymanik (Last author)