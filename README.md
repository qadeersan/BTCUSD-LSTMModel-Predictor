# Predicting Stock Prices Using Deep Learning for Sequences

### Project Abstract
   Stock price prediction is a challenging task due to the complex, dynamic, and noisy nature of financial markets. Accurate predictions of stock prices can provide valuable insights for investors and facilitate informed decision-making. The primary objective of this project is to develop a deep learning model for sequences that predicts daily stock prices using a comprehensive set of features, including price-based, volume-based, volatility-based, technical indicators, fundamental analysis, market-based, macroeconomic indicators, and time-based sequence shaped features.

   To achieve this goal, we will first extract and preprocess historical financial data from multiple sources including but not limited to public financial websites, APIs, and alternative data providers. We will then perform feature engineering to draw relevant information and create a wide range of features that capture different aspects of financial instrument price movements. We will also apply feature selection and dimensionality reduction techniques to identify the most significant features and avoid overfitting.

   For the methodology, we will focus on deep learning models for sequences, such as Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRU). These models are well suited for handling time series data and are capable of capturing the complex patterns dwelling in stock prices. We will train these models on a dataset split into training, validation, and testing sets. Model performance will be evaluated using multiple metrics including Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE). These will be used to compare the performance of different deep learning architectures to select the best one. 
   
   By the end of this project, we aim to create a reliable and robust stock prediction model that leverages the power of deep learning for sequences. With this we intend to provide valuable insights for investors and develop our understanding of quantitative finance and machine learning applications in the financial industry.

---

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for describing highlights for using this ML project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │   └── 1.0-mqa-data_analysis.ipynb  <-  Data correlation charts
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │   └── README.md      <- Youtube Video Link
    │   └── final_project_report <- final report .pdf format and supporting files
    │   └── presentation   <-  final power point presentation 
    |
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
       ├── __init__.py    <- Makes src a Python module
       │
       ├── preprocessing data           <- Scripts to download or generate data and pre-process the data
       │   └── make_dataset.py
       │   └── pre-processing.py
       │
       ├── features       <- Scripts to turn raw data into features for modeling
       │   └── build_features.py
       │
       ├── models         <- Scripts to train models and then use trained models to make
       │   │                 predictions
       │   ├── train_and_predict.py
       │   └── 
       │
       └── visualization  <- Scripts to create exploratory and results oriented visualizations
           └── visualize.py           

