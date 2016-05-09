# Recurrent Batch Normalization

This package provides a Chainer implementation of Batch-normalizing LSTM described in [Recurrent Batch Normalization [arXiv:1603.09025]](http://arxiv.org/abs/1603.09025).

[この記事](http://musyoku.github.io/2016/05/08/recurrent-batch-normalization/)で実装したコードです。

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