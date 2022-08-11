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
The embedding data can be obtained via running the *data_loader.py* file.


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
│   ├── embedding     # for demo  
│   └── raw
├── preprocessing
│   ├── data_loader     # for word embedding    
│   └── undersampling     
├── clusters                       
│   ├── BGL  # for demo
│       ├── ws=20
│       ├── ws=100
│       └── ws=200
│   ├── clustering     # use the obtained k for K-Means clustering; for each log seq, calculate the probability of being normal           
│   ├── clustering_prob     # combine the probability results for each log sequence   
│   ├── elbow_k     # generate the optimal k         
│   └── HDBSCAN_clustering  # the clustering method is applied in PLELog            
├── model
│   └── transformer_classification     # model training and prediction               
└── results_example                      
    └── BGL # for demo
```
*** 
- **Step 1:**
Create three folders with the corresponding DATASET_NAME (i.e., HDFS, BGL, and Thunderbird) under the directory *data/raw*, and download the raw datasets based on the above introduction, then put them into the created folders. 

- **Step2:** 
Run the *data_loader.py* for generating embeddings for each dataset, and put the generated files into the folder *data/embedding/DATASET_NAME*.

- **Step3:**
If you would like to conduct undersampling, please run the file *preprocessing/undersampling.py*. According to the self-defined undersampling rules, you can adjust the parameter *p* as the undersampling ratio of normal and abnormal logs.

- **Step4:** 
Run *clusters/elbow_k.py* to calculate the optimal two *k* values under the following ranges $k_1\in (11,15], k_2\in (35, 50]$.

- **Step5:** 
Run *clusters/clustering.py* and *clusters/clustering_prob.py* to get the label probability $P_{normal}$ for each log sequence.

- **Step6:** 
Run *model/transformer_classification.py* for model training and prediction.

*** 
**NOTE:**
In order to quickly implement CustomLog, we use BGL dataset as an example. 

1. By executing Step 1-3, the generated files can be founded in *data/embedding/BGL* folder. It contains the word embeddings of the training set with/without undersampling and the test set. The parameter *p* or *per* is defined as the undersampling ratio of normal and abnormal logs. For BGL, the values are 8 (ws=20), 7 (ws=100), and 6 (ws=200). 

2. The *k* values and the label probability are calculated, which can be obtained in *clusters/BGL* folder. 

3. For BGL, the corresponding pre-trained models with different window sizes (i.e., ws=20, 100, and 200) can be obtained here: [pre-trained models](https://figshare.com/articles/software/Pre-trained_model_for_BGL/20472333)
 
4. You only need to **run *model/transformer_classification.py*** for prediction, and the results should be the same as the following results.

## Results
![Results](https://github.com/ICSE2023/CustomLog/blob/main/figures/Results.png)
