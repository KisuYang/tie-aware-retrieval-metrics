# Tie-aware Retrieval Metrics (TRMs)

A lightweight Python library for reliable evaluation of retrieval systems in the presence of tied relevance scores.

When retrieval models operate in low numerical precision (e.g., BF16, FP16), many candidate documents receive identical scores, creating *spurious ties*. Conventional tie-oblivious evaluation arbitrarily breaks these ties, leading to unstable and potentially misleading metric values. *TRM* resolves this by computing *expected score* over all possible orderings of tied candidates, along with score *range* and *bias* diagnostics. *HPS* was omitted from this repo as its implementation is trivial.

**Paper:** [Reliable Evaluation Protocol for Low-Precision Retrieval](https://arxiv.org/abs/2508.03306) (ACL 2026)

## Installation

```bash
pip install tie-aware-retrieval-metrics  # being restored; install from source for now
```

Or install from source:

```bash
git clone https://github.com/KisuYang/tie-aware-retrieval-metrics.git
cd tie-aware-retrieval-metrics
pip install -e .
```

## Quick Start

```python
import trm

# Per-query relevance scores and labels
scores = [[0.99, 0.97, 0.97, 0.97, 0.95]]  # query 1: three docs share score 0.97

is_relevant = [[False, True, False, True, False]]  # query 1: docs 1, 3 are relevant

result = trm.evaluate(
    scores=scores,
    is_relevant=is_relevant,
    metrics=["ndcg", "mrr", "recall"],
    k_list=[3, 5])

# Macro-averaged results
for metric in ["ndcg", "mrr", "recall"]:
    for k in [3, 5]:
        r = result.metrics[metric][k]
        print(f"{metric}@{k}: E[M]={r.expected:.4f}  "
              f"M_obl={r.oblivious:.4f}  "
              f"M_max={r.maximum:.4f}  "
              f"M_min={r.minimum:.4f}  "
              f"range={r.range:.4f}  "
              f"bias={r.bias:.4f}")
```

Output:
```
ndcg@3: E[M]=0.4623  M_obl=0.3869  M_max=0.6934  M_min=0.3066  range=0.3869  bias=-0.0754
ndcg@5: E[M]=0.6383  M_obl=0.6509  M_max=0.6934  M_min=0.5706  range=0.1228  bias=0.0126
mrr@3:  E[M]=0.4444  M_obl=0.5000  M_max=0.5000  M_min=0.3333  range=0.1667  bias=0.0556
mrr@5:  E[M]=0.4444  M_obl=0.5000  M_max=0.5000  M_min=0.3333  range=0.1667  bias=0.0556
recall@3: E[M]=0.6667  M_obl=0.5000  M_max=1.0000  M_min=0.5000  range=0.5000  bias=-0.1667
recall@5: E[M]=1.0000  M_obl=1.0000  M_max=1.0000  M_min=1.0000  range=0.0000  bias=0.0000
```

## Supported Metrics

| Metric       | Key           | Paper Reference |
|-------------|---------------|-----------------|
| Hits@k      | `"hits"`      | Eq. 9           |
| Recall@k    | `"recall"`    | Eq. 10          |
| Precision@k | `"precision"` | Eq. 11          |
| F1@k        | `"f1"`        | Eq. 12          |
| nDCG@k      | `"ndcg"`      | Eq. 14-16       |
| MRR@k       | `"mrr"`       | Eq. 17-21       |
| MAP@k       | `"map"`       | Eq. 22-24       |

## API Reference

You can also selectively import:

```python
from trm import evaluate, build_tie_groups
```

#### `trm.evaluate(scores, is_relevant, metrics=None, k_list=None)`

Compute tie-aware retrieval metrics over a set of queries.

**Parameters:**
- `scores` (list of list of float): Per-query relevance scores for each candidate document.
- `is_relevant` (list of list of bool): Per-query binary relevance labels.
- `metrics` (list of str, optional): Metrics to compute. Supported: `"ndcg"`, `"mrr"`, `"map"`, `"recall"`, `"precision"`, `"f1"`, `"hits"`. Default: `["ndcg", "mrr", "map", "recall"]`.
- `k_list` (list of int, optional): Cutoff values. Default: `[1, 3, 5, 10, 20, 50, 100]`.

**Returns:** `EvaluationOutput` with:
- `.metrics[metric_name][k]` &rarr; `TieAwareResult` (macro-averaged)
- `.per_query[metric_name][k]` &rarr; list of per-query `TieAwareResult`
- `.to_dict()` &rarr; flat dictionary for logging

#### `TieAwareResult`

| Attribute    | Description                              |
|-------------|------------------------------------------|
| `.expected`  | E[M] &mdash; expected score over all tie orderings |
| `.oblivious` | M_obl &mdash; tie-oblivious (index-preserving) score |
| `.maximum`   | M_max &mdash; best-case score                    |
| `.minimum`   | M_min &mdash; worst-case score                   |
| `.range`     | M_max - M_min (Eq. 4)                   |
| `.bias`      | M_obl - E[M] (Eq. 5)                    |

#### `trm.build_tie_groups(scores, is_relevant)`

Build tie groups from raw scores and relevance labels.

**Returns:** list of `(group_size, num_relevant)` tuples sorted by descending score.

## Citation

```bibtex
@inproceedings{yang2026reliable,
    title     = {Reliable Evaluation Protocol for Low-Precision Retrieval},
    author    = {Yang, Kisu and Jang, Yoonna and Jang, Hwanseok and Choi, Kenneth and Augenstein, Isabelle and Lim, Heuiseok},
    booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
    year      = {2026},
}
```

## License

Apache License 2.0
