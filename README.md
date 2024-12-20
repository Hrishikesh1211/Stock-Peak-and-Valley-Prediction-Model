------------------
TO RUN A TRAINING:
------------------

Run it in Google Colab: https://colab.research.google.com/drive/1t_3BoC24JifFZN8K5w-kA53iu6QWEhyf?usp=drive_link 

 

OR in a similar environment with: 

 

TensorFlow 2.17.1 

Pandas version: 2.2.2  

NumPy version: 1.26.4  

TensorFlow version: 2.17.1  

Scikit-learn version: 1.5.2  

Matplotlib version: 3.8.0  

Seaborn version: 0.13.2  



Then download and insert the necessary data files into ./content/data or adjust file paths in 'train and eval.py' to the data file locations

sequential_features_normalized_window12.csv  

sequential_features_normalized_window12_val.csv  

static_features_normalized_window12.csv  

static_features_normalized_window12_val.csv  

labels_interval_5_window_12_train.csv  

labels_interval_5_window_12_val.csv 

 

Run 'train and eval.py'


------------------------
TO RUN THE DEMONSTRATION:
------------------------

Ensure that all the necessary files are downloaded so that the file path can be easily obtained. The necessary files are as follows: 

my_model.keras  

sequential_features_normalized_window12_val.csv  

static_features_normalized_window12_val.csv  

labels_interval_5_window_12_val.csv  

data_5.csv 

StockPrediction_Demo.py 

Open StockPrediction_Demo.py and Update the values of MODEL_PATH, SEQ_VAL_FILE, STATIC_VAL_FILE, LABEL_VAL_FILE, and DATA_5_PATH to match the file path of the corresponding files saved previously. 

Run the StockPrediction_Demo.py file on any virtual environment that has access to the following required libraries: 

PyQt5 

Tensorflow 2.17.1 or above and dependencies (Will NOT work on 2.10, the default Conda version)

Pandas, Numpy, nltk, requests and csv 
