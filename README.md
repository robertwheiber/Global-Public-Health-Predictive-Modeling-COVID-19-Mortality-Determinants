# Global-Public-Health-Predictive-Modeling-COVID-19-Mortality-Determinants
Predictive Covid-19 Mortality Determinants

# The Analytical Problem
During the COVID-19 pandemic, mortality outcomes varied drastically across different nations, creating a complex web of variables driven by socioeconomic status, healthcare capacity, and policy responses. This project seeks to build a predictive machine learning model to identify the most significant determinants of total death rates, segmenting the global response into "rich" and "poor" countries based on GDP per capita to isolate differing risk factors.

# The Tech Stack

o  Language: Python

o  Machine Learning: Scikit-Learn (Decision Tree Regressors)

o  Data Engineering: Pandas, Numpy (Data imputation, subsetting by GDP)

o  Data Visualization: Matplotlib, Seaborn (Correlation heatmaps, EDA)

# Methodology

1  Data Engineering & Imputation: Acquired comprehensive daily-level COVID-19 data from Our World in Data (covering 56 countries across 6 continents). Handled missing independent variables via statistical imputation to ensure robust model training.

2  Exploratory Data Analysis (EDA): Generated correlation heatmaps to identify baseline associations between total deaths and healthcare system capacities, policy stringency, and demographics.

3  Machine Learning Architecture: Split the dataset into "rich" and "poor" categories and engineered separate Decision Tree models (using an 80/20 train/test split) to map features to the target outcome (total deaths).

4  Feature Importance Extraction: Analyzed the decision tree nodes to quantify the percentage of predictive power each variable held within its respective economic bracket.

# Core Implementation Highlights

# 1  Decision Tree Architecture & Evaluation Pipeline
Architected a reusable training pipeline utilizing Scikit-Learn's DecisionTreeRegressor. Applied strict hyperparameters (max_depth, min_samples_leaf) to prevent overfitting on the complex health datasets.

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def fit_and_evaluate_tree(data, group_label, feature_list, target_col):
    # Execute 80/20 Train-Test Split
    X_train, X_test, y_train, y_test = split_xy(data, feature_list, target_col)

    # Instantiate Decision Tree with overfitting constraints
    tree_reg = DecisionTreeRegressor(
        max_depth=5,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42
    )
    tree_reg.fit(X_train, y_train)

    # Evaluate Model Accuracy
    y_pred = tree_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"[{group_label}] RMSE: {rmse:.2f} | MAE: {mae:.2f} | R-squared: {r2:.4f}")
    
    return tree_reg
# 2. Feature Importance Extraction
Extracted the internal node weights from the trained model to quantify exactly which variables drove the most variance in mortality, transforming a "black box" model into an actionable business narrative

import pandas as pd

def extract_feature_importance(trained_model, feature_list):
    # Map feature importances to their respective column names
    importances = pd.Series(trained_model.feature_importances_, index=feature_list)
    
    print('\nFeature Importances (Predictive Power):')
    print(importances.sort_values(ascending=False))
    
    return importances

# Execute extraction for the deployed model
poor_country_importances = extract_feature_importance(tree_reg, feature_list)

# 3. Global Sub-Population Modeling & Transformation
Utilized Column Transformers to dynamically handle categorical string variables (like "Rich/Poor" segmentations) across the global dataset while passing numerical features through cleanly.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#Define features and target mapping
X = df_model[features + ['Rich/Poor']]
y = df_model[target]

#Construct Pipeline to encode categorical data
preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), ['Rich/Poor']),
        ('num', 'passthrough', features)
    ]
)

#Apply transformation prior to fitting the Global Decision Tree
X_transformed = preprocess.fit_transform(X)

# Impact & Results

o  Poor Countries Model: Identified that underlying health factors—specifically the cardiovascular death rate (accounting for 72% of predictive power) and hospital beds per thousand (21%)—were the dominant predictors of mortality.

o  Rich Countries Model: Revealed a fundamentally different pattern, with population density (77.5%) and ICU patient rates (16.7%) overwhelmingly driving mortality outcomes, demonstrating that transmission density outweighed baseline health factors in developed nations.

o  High Predictive Accuracy: The models achieved exceptionally high R² values (0.98–0.99), highlighting the models' ability to capture the variance in the target data.

