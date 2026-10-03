# Memory Construction Evaluation

候选数：30

| strategy | precision | recall | F1 | saved | tokens |
|---|---:|---:|---:|---:|---:|
| store_all | 0.700 | 1.000 | 0.824 | 30 | 190 |
| heuristic | 0.870 | 0.952 | 0.909 | 23 | 168 |
| explicit_only | 1.000 | 0.143 | 0.250 | 3 | 27 |
| quality_aware | 0.857 | 0.571 | 0.686 | 14 | 97 |

数据集是可复现的开发/演示标注集，不代表独立人工 benchmark。Token 为确定性估算，不是供应商 tokenizer 的真实计费值。
