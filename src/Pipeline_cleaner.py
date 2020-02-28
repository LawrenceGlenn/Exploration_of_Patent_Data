import pandas as pd
def remove_columns(df,cols = []):
    if len(cols)==0:
        cols= ["id",'abstract','title','date']
    return df.drop(cols,axis=1)

def make_dummies(df):
    return pd.get_dummies(df)