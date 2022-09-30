import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# create a function that return a list of data sets and relevant information
###
### @input: df (pd data frame), features (features of interest), target_var (str)
### @output: a list of train and test sets

### df - 2018 CA data from acs_data = data_source.get_data(states=["CA"], download=True)
### features - features from features, label, _ = ACSEmployment.df_to_pandas(acs_data)
### target_var - "WAGP"

def preProcessor(df, features, target_var):
    copy = features.copy()
    
    # handle possible NaN values
    copy[target_var] = df[target_var]
    copy = copy.dropna().reset_index(drop=True)
    
    # choose top sensitive races
    copy = copy[(copy["RAC1P"] == 1) | (copy["RAC1P"] == 2.0) | (copy["RAC1P"] == 6) | (copy["RAC1P"] == 8)]
    
    # seperate the data
    X, y = copy.iloc[:, :16], copy[target_var]
    
    # standerdize the data
    scalerX, scalerY = StandardScaler().fit(X), StandardScaler().fit(np.array(y).reshape(-1, 1))
    X, y = scalerX.transform(X), scalerY.transform(np.array(y).reshape(-1, 1))
    
    # create the testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return [X_train, X_test, y_train, y_test]
    

if __name__ == "__main__":
    preProcessor(df, features, target_var)