import pandas as pd
from src.Clean_Abstracts import *
from src.Tfidf_on_Pandas import *

def remove_columns(df,cols = []):
    if len(cols)==0:
        cols= ["id",'patent_id','abstract','title','date','Unnamed: 0','Unnamed: 0.1','abstract_cleaned']
    return df.drop(cols,axis=1)

def make_dummies(df):
    return pd.get_dummies(df, drop_first=True, columns=['type', 'kind','field_id'])
    
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
        

def add_wipo(df):
    wipo = pd.read_csv("data/wipo.tsv", sep="\t")
    wipo['patent_id'] = wipo['patent_id'].astype('str')
    wipo['field_id'] = wipo['field_id'].astype('str')
    return df.merge(wipo, left_on='id', right_on='patent_id')
    
def tfidf_abstract(train,test):

    train['train']=1
    test['train']=0
    combined = pd.concat([train,test])
    combined.loc[combined['abstract'].isna(), 'abstract'] = ""
    combined = clean_and_abstract_pd(combined)
    combined = lem_abstract_pd(combined)
    combined = stem_abstract_pd(combined)
    tfidf_mat = tfidf_vector_matrix(combined,2,3, col='abstract_cleaned' )
    combined = pd.concat([combined.reset_index(), tfidf_mat.reset_index()], axis=1, sort=False)
    combined.drop(['index'],axis=1,inplace=True)
    train_df = combined[combined["train"] == 1]
    test_df = combined[combined["train"] == 0]
    train_df.drop(['train'],axis=1,inplace=True)
    test_df.drop(['train'],axis=1,inplace=True)
    return train_df,test_df