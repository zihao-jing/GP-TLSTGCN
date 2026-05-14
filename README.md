# EC-Prune: Data-Centric Graph Pruning for Spatiotemporal Foundation Model Adaptation

*ICML 2026 Workshop Submission*

## Abstract

Cross-domain transfer of spatiotemporal graph models is frequently impaired by topology mismatch and boundary-driven noise: boundary sensors in road networks are influenced by external regions not represented in the modeled graph, injecting spurious patterns that inflate domain discrepancy and degrade out-of-distribution performance. We propose **EC-Prune**, a model-agnostic, data-centric graph pruning module that serves as a preprocessing step for any spatiotemporal graph model. EC-Prune applies an entropy–correlation dual-criteria score to identify and remove outer-layer nodes with weak intra-graph support, yielding a compact, boundary-denoised subgraph that reduces cross-domain distributional shift. Operating entirely on the input data rather than model internals, EC-Prune requires no architectural modification and enhances the cross-domain transferability of spatiotemporal foundation models. Evaluated across five modern graph forecasting baselines on standard traffic benchmarks under limited target data, EC-Prune achieves an average gain of **14.1%** across all metrics.

## Method Overview

EC-Prune is a two-stage preprocessing pipeline:

1. **Information Entropy Analyzer (IEA)** — assigns a scalar score to every edge by combining Shannon entropy of its endpoint nodes with their pairwise Pearson correlation:

   - Node entropy: $H_i = -\sum_b p_i(b) \log(p_i(b) + \epsilon)$
   - Pairwise correlation: $r_{ij}$ (absolute Pearson)
   - Edge score: $s_{ij} = A_{ij} \cdot r_{ij} \cdot \frac{H_i + H_j}{2}$

   Boundary nodes driven by out-of-graph factors have low correlation with in-graph neighbors, causing their edges to score low.

2. **Outer-Layer Graph Pruning (GP)** — thresholds edge scores and iteratively removes nodes with insufficient intra-graph support, peeling off boundary rings to yield a compact, self-consistent subgraph $\mathcal{G}' = (\mathcal{V}', \mathcal{E}')$.

The same pipeline is applied independently to source and target domains, reducing cross-domain distributional shift for any downstream model without requiring architectural changes.

## Datasets

### Sources
1. METR-LA: [DCRNN author's Google Drive](https://drive.google.com/file/d/1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC/view?usp=sharing)
2. PEMS-BAY: [DCRNN author's Google Drive](https://drive.google.com/file/d/1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq/view?usp=sharing)
3. PEMSD7-M: [STGCN author's GitHub repository](https://github.com/VeritasYin/STGCN_IJCAI-18/blob/master/data_loader/PeMS-M.zip)

### Preprocessing
We follow the ChebNet formulation for graph preprocessing:

<img src="./figure/weighted_adjacency_matrix.png" style="zoom:100%" />

**Transfer setup**: pretrain on METR-LA, adapt to PEMSD7-M and PEMS-BAY using only **10%** of the target-domain training data.

## Results

EC-Prune is evaluated as a plug-and-play augmentation on five spatiotemporal graph baselines. All other hyperparameters are held fixed.

| Model | Aug | P7M MAE@15 | P7M MAPE@15 | P7M RMSE@15 | PB MAE@15 | Avg. Gain |
|---|---|---|---|---|---|---|
| STGCN | base | 3.24 | 5.65 | 5.27 | 4.17 | — |
| STGCN | +EC-Prune | 3.09 | 5.29 | 5.17 | 3.58 | 7.4% |
| DCRNN | base | 3.40 | 5.77 | 5.74 | 4.73 | — |
| DCRNN | +EC-Prune | 3.17 | 5.57 | 5.59 | 3.63 | 15.8% |
| Graph WaveNet | base | 2.82 | 4.92 | 4.68 | 4.34 | — |
| Graph WaveNet | +EC-Prune | 2.61 | 4.41 | 4.49 | 3.96 | 8.7% |
| ASTGNN | base | 2.77 | 4.88 | 4.75 | 4.04 | — |
| ASTGNN | +EC-Prune | 2.44 | 4.21 | 4.40 | 3.07 | 17.3% |
| PDFormer | base | 2.67 | 4.74 | 4.63 | 4.78 | — |
| PDFormer | +EC-Prune | 2.58 | 4.46 | 4.45 | 3.06 | 21.2% |

See `paper_repo/csv_tables/` for full results at 15- and 30-minute horizons on both target datasets.

### Ablation Study

Using STGCN on METR-LA → PEMSD7-M with 10% target data:

| Variant | MAE@15 | RMSE@15 | MAE@30 | RMSE@30 |
|---|---|---|---|---|
| (i) No pruning | 3.24 | 5.27 | 4.48 | 7.50 |
| (ii) Correlation-only | 3.19 | 5.28 | 4.55 | 7.55 |
| (iii) Entropy-only | 3.14 | 5.38 | 4.54 | 7.47 |
| (iv) Score threshold only (no OL removal) | 3.10 | 5.44 | 4.54 | 7.46 |
| **(v) EC-Prune (full)** | **3.09** | **5.17** | **4.37** | **7.40** |

Both criteria and outer-layer removal are necessary; omitting either degrades performance.

## Experimental Setup

- **Hardware**: 2× H100-80GB GPUs
- **Split**: 7:1.5:1.5 train/val/test
- **Training**: batch size 32, lr 1e-3, weight decay 5e-4, early stopping (max 200 epochs)
- **History length**: 12 steps (24 steps for PEMS-BAY 30-min horizon)
- **Prediction horizons**: 15 min and 30 min
- **Metrics**: MAE, MAPE (%), RMSE

## Related Works

1. STGCN: [*Spatio-Temporal Graph Convolutional Networks*](https://arxiv.org/abs/1709.04875)
2. DCRNN: [*Diffusion Convolutional Recurrent Neural Network*](https://arxiv.org/abs/1707.01926)
3. Graph WaveNet: [*Adaptive Graph Convolutional Recurrent Network*](https://arxiv.org/abs/1906.00121)
4. PDFormer: [*Propagation Delay-aware Dynamic Long-range Transformer*](https://arxiv.org/abs/2301.07945)
5. ChebNet: [*Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering*](https://arxiv.org/abs/1606.09375)

## Related Code

1. STGCN: https://github.com/VeritasYin/STGCN_IJCAI-18
2. DCRNN: https://github.com/liyaguang/DCRNN
3. Graph WaveNet: https://github.com/nnzhan/Graph-WaveNet
4. ChebNet: https://github.com/mdeff/cnn_graph

## Implementation Notes

- EC-Prune is model-agnostic: apply it as a preprocessing step before any backbone model.
- Provides separate configs for ChebyGraphConv and GraphConv.
- Includes METR-LA, PEMS-BAY, and PEMSD7-M datasets with updated preprocessing.
- Early stopping and dropout are used for stable training.

## Requirements

```console
pip3 install -r requirements.txt
```
