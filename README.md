# Scaling Causal Inference in Additive Noise Models: KIKO (Knock In Knock Out): 

Link to the paper: http://proceedings.mlr.press/v104/assaad19a/assaad19a.pdf

Required packages:
* tensorflow
* numpy
* sklearn
* scipy
* joblib

In order to use KIKO for causal discovery on the simulated data available at "tools/generator.py" execute kiko.py with the desired options (arg1 refers to number of variable and arg2 refers to number of observations):
```shell
python3 kiko.py 4 500
```

If number of variables is 4 an error measure will be calculated since we have access to the true DAG. Otherwise the algorithm will be executed without validation (in this case, the purpose is to check execution time).
More hyperparameters can be modified in kiko.py such as activation functions, number of neurones, noise ...

In order to use KIKO on the benchmark datasets use benchmark_causality_pairs.py
