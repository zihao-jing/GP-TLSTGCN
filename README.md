# Pruning for Generalization: A Transfer-Oriented Spatiotemporal Graph Framework

## Abstract
Multivariate time series forecasting in graph-structured domains is critical for real-world applications, yet existing spatiotemporal models often suffer from performance degradation under data scarcity and cross-domain shifts. We address these challenges through the lens of structure-aware context selection. We propose **TL-GPSTGN**, a transfer-oriented spatiotemporal framework that enhances sample efficiency and out-of-distribution generalization by selectively pruning non-optimized graph context. Specifically, our method employs information-theoretic and correlation-based criteria to extract structurally informative subgraphs and features, resulting in a compact, semantically grounded representation. This optimized context is subsequently integrated into a spatiotemporal convolutional architecture to capture complex multivariate dynamics. Evaluations on large-scale traffic benchmarks demonstrate that TL-GPSTGN consistently outperforms baselines in both zero-shot and low-data transfer scenarios. Our findings suggest that explicit context pruning serves as a powerful inductive bias for improving the robustness of graph-based forecasting models.

## TL-GPSTGN at a glance
- **Structure-aware pruning** selects informative subgraphs and features.
- **Compact context** improves sample efficiency and cross-domain transfer.
- **Spatiotemporal convolution** captures multivariate dynamics with optimized graph context.

## Datasets
### Sources
1. METR-LA: [DCRNN author's Google Drive](https://drive.google.com/file/d/1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC/view?usp=sharing)
2. PEMS-BAY: [DCRNN author's Google Drive](https://drive.google.com/file/d/1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq/view?usp=sharing)
3. PeMSD7(M): [STGCN author's GitHub repository](https://github.com/VeritasYin/STGCN_IJCAI-18/blob/master/data_loader/PeMS-M.zip)

### Preprocessing
We follow the ChebNet formulation for graph preprocessing:
<img src="./figure/weighted_adjacency_matrix.png" style="zoom:100%" />

## Related works
1. TCN: [*An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*](https://arxiv.org/abs/1803.01271)
2. GLU and GTU: [*Language Modeling with Gated Convolutional Networks*](https://arxiv.org/abs/1612.08083)
3. ChebNet: [*Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering*](https://arxiv.org/abs/1606.09375)
4. GCN: [*Semi-Supervised Classification with Graph Convolutional Networks*](https://arxiv.org/abs/1609.02907)

## Related code
1. TCN: https://github.com/locuslab/TCN
2. ChebNet: https://github.com/mdeff/cnn_graph
3. GCN: https://github.com/tkipf/pygcn

## Implementation notes
- Adds early stopping and dropout for stable training.
- Provides separate configs for ChebyGraphConv and GraphConv.
- Includes METR-LA and PEMS-BAY datasets and updated preprocessing.

## Requirements
```console
pip3 install -r requirements.txt
```
