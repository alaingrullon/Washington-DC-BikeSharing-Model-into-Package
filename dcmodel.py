"""
The Washington DC Biking prediction model package. It predicts how many bikers there are at a given future moment in time 
given some features.
"""

__version__ = "0.1.6"

# imports at the beginning!
import pandas as pd
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.pipeline import FeatureUnion, make_union
from sklearn.ensemble import RandomForestRegressor
import black

# Functions used inside our train_and_persist function

# Imputation purposes
def ffill_missing(ser):
    return ser.fillna(method="ffill")

# Feature Engineering (weekend)
def is_weekend(data):
    return (
        data["dteday"]
        .dt.day_name()
        .isin(["Saturday", "Sunday"])
        .to_frame()
    )

# Feature Engineering (year)
def year(data):
    # Our reference year is 2011, the beginning of the training dataset
    return (data["dteday"].dt.year - 2011).to_frame()

# Train the model and persist
def train_and_persist():
    # Read in data into a Pandas DataFrame
    df = pd.read_csv("hour.csv", parse_dates=["dteday"])
    
    # Assign features to independent (X) and predicted (y) variables 
    X = df.drop(columns=["instant", "cnt", "casual", "registered"])
    y = df["cnt"]
    
    # using ffill_missing to make imputter with forward fill
    ffiller = FunctionTransformer(ffill_missing)
    
    # Make weather imputter pipeline for later use 
    weather_enc = make_pipeline(
        ffiller,
        OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=X["weathersit"].nunique()
        ),
    )
    
    # Make column transformer for imputtation and encoding process
    ct = make_column_transformer(
        (ffiller, make_column_selector(dtype_include=np.number)),
        (weather_enc, ["weathersit"]),
    )
    
    # Make preprocessing object for Feature Engineering 
    preprocessing = FeatureUnion([
        ("is_weekend", FunctionTransformer(is_weekend)),
        ("year", FunctionTransformer(year)),
        ("column_transform", ct)
    ])
    
    # Define Pipeline to separate preprocessing and modelling
    reg = Pipeline([
        ("preprocessing", preprocessing), 
        ("model", RandomForestRegressor())
    ])
    
    # Train, test split: Train is before 10/2012 and Test is after 10/2012
    X_train, y_train = X.loc[X["dteday"] < "2012-10"], y.loc[X["dteday"] < "2012-10"]
    
    X_test, y_test = X.loc["2012-10" <= X["dteday"]], y.loc["2012-10" <= X["dteday"]]
    
    # Train the model
    reg.fit(X_train, y_train)
    
#     # Evaluate to get R-squared
#     reg.score(X_test, y_test)
    
#     # Predict 
#     y_pred = reg.predict(X_test)
    
    # Create the joblib file
    joblib.dump(reg, "biking.joblib")

    print("Model trained successfully")
    
# Predict number of bikers given a user input
def predict(dteday, hr, weathersit, temp, atemp, hum, windspeed):
    # Load in the joblib created by train_and_persist
    try:
        reg = joblib.load("biking.joblib")
    
    # Predict number of bikers based on the variables
        y_pred = reg.predict(pd.DataFrame([[
                        pd.to_datetime(dteday),
                        hr,
                        weathersit,
                        temp,
                        atemp,
                        hum,
                        windspeed,
                    ]], columns=[
                                'dteday',
                                'hr',
                                'weathersit',
                                'temp',
                                'atemp',
                                'hum',
                                'windspeed'
                    ]))
        
    except: 
        train_and_persist()
        
    if y_pred[0] >= 0:
        return y_pred[0]
    else:
        print("Invalid inputs. Must be positive values.")

    
    
    
    
    
    
    
    
    
    

