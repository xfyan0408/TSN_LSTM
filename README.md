## Resource Predictor

This project follows `doc/实现框架.md`.

```powershell
cd predictor
python train.py
python infer.py --out pred.csv
```

Main files:

- `predictor/config.py`: hyperparameters and paths
- `predictor/dataset.py`: CSV loading, normalization, sliding windows
- `predictor/model.py`: ChannelMix + TCN + LSTM + three heads
- `predictor/train.py`: training entrypoint
- `predictor/infer.py`: inference entrypoint
- `predictor/utils.py`: checkpoint and metrics helpers
