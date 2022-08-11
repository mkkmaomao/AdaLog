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
│   └── raw
├── preprocessing
│   ├── data_loader
│   └── undersampling
├── clusters                       
│   ├── BGL
│       ├── ws=20
│       ├── ws=100
│       └── ws=200
│   ├── clustering                
│   ├── clustering_prob         
│   ├── elbow_k                
│   └── HDBSCAN_clustering              
├── model
│   └── transformer_classification               
├── results_example                      
│   └── BGL

