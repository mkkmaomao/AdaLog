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
The raw data has been found here: [HDFS](https://figshare.com/articles/dataset/HDFS/20472282), [BGL](https://figshare.com/articles/dataset/BGL/20472270), and [Thunderbird](https://figshare.com/articles/dataset/Thunderbird/20472297).

Use the BGL dataset as an example, the corresponding pre-trained model with different window size (i.e., ws=20, 100, and 200) can be obtained: [pre-trained models](https://figshare.com/articles/software/Pre-trained_model_for_BGL/20472333)

## Reproducibility
### Requirements
Python 3.6+

tensorflow-gpu 2.4

transformers

tf-models-official 2.4.0

scikit-learn

pandas

numpy

gensim

keras 

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
└── results_example                      
    └── BGL
.
***

