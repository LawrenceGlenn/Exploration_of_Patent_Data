import pandas as pd
def remove_columns(df,cols = []):
    if len(cols)==0:
        cols= ["id",'abstract','title','date','type','Unnamed: 0','Unnamed: 0.1']
    return df.drop(cols,axis=1)

def make_dummies(df):
    return pd.get_dummies(df, drop_first=True)

def clean(train,test):
    train['train']=1
    test['train']=0
    combined = pd.concat([train,test])
    combined =remove_columns(combined)
    combined = combined.fillna(0)
    combined = make_dummies(combined)
    train_df = combined[combined["train"] == 1]
    test_df = combined[combined["train"] == 0]
    train_df.drop(['train'],axis=1,inplace=True)
    test_df.drop(['train'],axis=1,inplace=True)
    return train_df,test_df