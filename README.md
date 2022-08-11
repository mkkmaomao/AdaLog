# CustomLog
This is the basic implementation of CustomLog in ICSE 2023.
- CustomLog
  - Abstract
  - Framwork
  - Datasets
  - Reproducibility
    - Requirements
    - Anomaly Detection
    - Results

## Abstract

## Framework
![Framework](https://github.com/ICSE2023/CustomLog/blob/main/figures/Framework.png)

## Datasets

## Reproducibility
### Requirements
### Anomaly Detection
```
.
├── data                   
│   ├── embedding
│   ├── raw
├── preprocessing
│   │   ├── data_loader
│   │   └── undersampling
├── clusters                       // 应用
│   ├── BGL
│   │   ├── ws=20
│   │   ├── ws=100
│   │   └── ws=200
│   ├── clustering                // 开发环境
│   ├── clustering_prob         // 实验
│   ├── elbow_k                // 配置控制
│   └── HDBSCAN_clustering              // 本地
├── model
│   └── transformer_classification               // 测试环境

