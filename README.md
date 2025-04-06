# Random Forest Project - Income Prediction

## Overview
This project uses ensemble machine learning techniques, specifically Random Forests, to predict whether an individual's income exceeds $50,000 per year based on census data. The project demonstrates the full machine learning workflow including data preprocessing, model building, hyperparameter tuning, and model evaluation.

## Dataset
The dataset used is the UCI Adult Census Income dataset, which contains demographic information about individuals, including:
- Age
- Workclass
- Education
- Marital status
- Occupation
- Race
- Sex
- Hours worked per week
- Native country
- Income (target variable: >50K or <=50K)

## Project Structure
```
Random_Forest_Project/ 
    │ adult.data # UCI Census dataset 
    │ feature_importances_first_model.csv # Feature importance rankings for initial model 
    │ feature_importances_second_model.csv # Feature importance rankings for improved model 
    │ README.md # This file 
    │ script.py # Main Python script for the project
```


## Methodology
1. **Data Preprocessing**
   - Cleaned categorical data by removing leading whitespace
   - Converted categorical variables to dummy variables
   - Created binary target variable from income data

2. **Initial Model & Baseline**
   - Built a Random Forest classifier with default parameters
   - Evaluated baseline performance with accuracy, precision, recall, F1 and ROC AUC metrics

3. **Hyperparameter Tuning**
   - Tuned the `max_depth` parameter over a range from 1 to 25
   - Selected model with best test accuracy
   - Analyzed feature importances

4. **Feature Engineering**
   - Created a binned education feature (`education_bin`)
   - Added a binary US citizenship feature (`us_citizen`)

5. **Model Improvement**
   - Retrained the model with the new feature set
   - Retuned hyperparameters to find optimal model
   - Compared performance with the first model

6. **Extended Tuning**
   - Further tuned `n_estimators` parameter
   - Experimented with different `max_features` settings
   - Built final optimized model

## Key Results
- Successfully built a Random Forest model that predicts income level with high accuracy
- Identified the most important features for income prediction
- Showed how feature engineering can improve model performance

## Top Features for Income Prediction
First model top features included:
- Capital gain
- Marital status (being married)
- Age
- Education level
- Hours worked per week

Second model (with engineered features) top features:
- Capital gain
- Age
- Hours worked per week
- Higher education (Masters or more)
- Capital loss

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Usage
To run the project:
```bash
cd Random_Forest_Project
python script.py
```