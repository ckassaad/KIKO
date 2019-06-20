# KIKO (Knock In Knock Out): Scaling Causal Inference in Additive Noise Models

Link to the paper: soon

Required packages:
* tensorflow
* numpy
* sklearn
* scipy
* joblib

In order to use KIKO for causal discovery of simulated data use kiko.py with the desired option.
```python
import pandas as pd
from sam import SAM
sam = SAM()
data = pd.read_csv("test/G5_v1_numdata.tab", sep="\t")
output = sam.predict(data, nruns=12) # Recommended if high computational capability available, else nruns=1
```

In order use KIKO on the benchmark datasets use benchmark_causality_pairs.py
