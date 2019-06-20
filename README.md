# KIKO (Knock In Knock Out): Scaling Causal Inference in Additive Noise Models

Link to the paper: soon

Required packages:
* tensorflow
* numpy
* sklearn
* scipy
* joblib

In order to use KIKO for causal discovery on simulated data execute kiko.py with the desired option (number of variable and number of observations).
```shell
python kiko.py 4 500
```
If number of variable is 4 a error measure will be calculated since we have access to the true DAG. Otherwise the algorithm will be executed without validation (in this case, the purpose is to check execution time).

In order use KIKO on the benchmark datasets use benchmark_causality_pairs.py
