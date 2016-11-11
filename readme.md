# Recurrent Batch Normalization

This package provides a Chainer implementation of Batch-normalizing LSTM described in [Recurrent Batch Normalization [arXiv:1603.09025]](http://arxiv.org/abs/1603.09025).

#### Todo:
- [ ] separate statistics

```
The batch normalization transform relies on batch statistics to standardize the LSTM activations. It would seem natural to share the statistics that are used for normalization across time, just as recurrent neural networks share their parameters over time. However, we have found that simply averaging statistics over time severely degrades performance. Although LSTM activations do converge to a stationary distribution, we have empirically observed that their statistics during the initial transient differ significantly as figure 1 shows. Consequently, we recommend using separate statistics for each timestep to preserve information of the initial transient phase in the activations.
```

## Requirements

- Chainer 1.8+

## Running

Before

```
from chainer import links as L
lstm = L.LSTM(n_in, n_out)
```

After

```
from bnlstm import BNLSTM
lstm = BNLSTM(n_in, n_out)
```