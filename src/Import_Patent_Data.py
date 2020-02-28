import pyspark as ps    # for the pyspark suite
import pandas as pd
from pyspark.sql import Window
from pyspark.sql.functions import *
import itertools as IT
import random
from sklearn.model_selection import train_test_split

def instantiate_spark():
	spark = (ps.sql.SparkSession
         .builder
         .master('local[*]')
         .appName('lecture')
         .getOrCreate()
        )
	return spark

def import_data():
	spark = instantiate_spark()

	df = spark.read.option("inferschema", "true").csv('data/patent.tsv',
                    header=True,       # use headers or not
                    quote='"',         # char for quotes
                    sep="\t")          # char for separation  
	df = df.drop('country')
	df = df.drop('filename')
	df = df.drop('number')

	df_cpc = spark.read.option("inferschema", "true").csv('data/cpc_current.tsv',
                    header=True,       # use headers or not
                    quote='"',         # char for quotes
                    sep="\t")          # char for separation  

	df_with_groups = df.join(df_cpc, df.id == df_cpc.patent_id).drop('id')
	
	
	w = Window.partitionBy('patent_id')
	df_with_groups = df_with_groups.withColumn('maxB', max('sequence').over(w))\
    	.where(col('sequence') == col('maxB'))\
    	.drop('maxB')\
    	.sort(col("patent_id"))

	return df_with_groups

def import_ptab_data():
    spark = instantiate_spark()

    df = spark.read.option("inferschema", "true").csv('data/PTAB_AIA_Trials.csv',
                    header=True,       # use headers or not
                    quote='"',         # char for quotes
                    sep="|")          # char for separation
    return df
    

def limit_by_year(df, year='2012-01-01 00:00:00'):
    
    return df.filter(col("date") >= unix_timestamp(lit(year)).cast('timestamp'))

def create_boolean_aia_trial_col(df,ptab):
    df3 = df.withColumn("id", col("patent_id")).alias("df2")\
        .join(ptab.withColumn("id", col("respondentPatentNumber")).alias("bears2"), on="id", how="left")\
        .select("df2.*", when(col("bears2.respondentPatentNumber").isNotNull(), 1).otherwise(0).alias("AIA"))
    return df3

def import_data_pd(min_date = "1900", max_date = "3000"):
    chunksize = 10 ** 5
    chunks = pd.read_csv('data/patent.tsv', chunksize=chunksize, sep="\t")
    df = pd.concat(valid_chunks(chunks, min_date, max_date))
    df = df.drop('country', axis=1)
    df = df.drop('filename', axis=1)
    df = df.drop('number', axis=1)
#    df = df.dropna()
#    ptab = pd.read_csv("data/PTAB_AIA_Trials.csv", sep="|")
#    df_cpc = pd.read_csv('data/cpc_current.tsv', sep="\t")
#    df_cpc = df_cpc.dropna()
#    g = df_cpc.groupby(['patent_id'])['sequence'].transform('max')
#    df_cpc = df_cpc[(df_cpc['sequence'] == g)]
#    new_df = pd.merge(left=df,right=df_cpc, left_on='id', right_on='patent_id')
#    new_df = new_df.drop('id', axis=1)
    return df


def create_aia_patent_pd_files(min_date,max_date):
    df = import_data_pd(min_date,max_date)
    ptab = pd.read_csv("data/PTAB_AIA_Trials.csv", sep="|")
    df['aia'] = df["id"].astype(str).isin(ptab['respondentPatentNumber']).astype(int)
    df.to_csv("data/temp/patents_with_aia_"+min_date+"_"+max_date+".csv", sep="|")
    
def load_aia_patents(value=1, sample=1):
    files = ['data/temp/patents_with_aia_1900_2014.csv','data/temp/patents_with_aia_2014_2018.csv',
            'data/temp/patents_with_aia_2018_2019.csv']
    result = pd.DataFrame()
    for fl in files:
        df = pd.read_csv(fl, sep="|")
        df = df[df['aia']==value]
        df = df.sample(frac=sample, replace=False, random_state=1)
        result = pd.concat([result,df])
    
    return result


    
def import_aia_patent_data_pd():
    ptab = pd.read_csv("data/PTAB_AIA_Trials.csv", sep="|")
    chunksize = 10 ** 5
    chunks = pd.read_csv('data/patent.tsv', chunksize=chunksize, sep="\t")
    df = pd.concat(valid_chunks(chunks, min_date))
    df = df.drop('country', axis=1)
    df = df.drop('filename', axis=1)
    df = df.drop('number', axis=1)
#    df = df.dropna()
#    ptab = pd.read_csv("data/PTAB_AIA_Trials.csv", sep="|")
#    df_cpc = pd.read_csv('data/cpc_current.tsv', sep="\t")
#    df_cpc = df_cpc.dropna()
#    g = df_cpc.groupby(['patent_id'])['sequence'].transform('max')
#    df_cpc = df_cpc[(df_cpc['sequence'] == g)]
#    new_df = pd.merge(left=df,right=df_cpc, left_on='id', right_on='patent_id')
#    new_df = new_df.drop('id', axis=1)
    return df



def valid_chunks(chunks, min_date, max_date):
    for chunk in chunks:
        mask = (chunk['date'] <= max_date) & (chunk['date'] > min_date)
        if mask.all():
            yield chunk
        else:
            yield chunk.loc[mask]
            



def create_train_and_test(ratio = .8):
    df = import_data()
    ptab = import_ptab_data()
    df = limit_by_year(df)
    df_with_aia = create_boolean_aia_trial_col(df,ptab)
    train,test = df_with_aia.randomSplit([ratio, 1-ratio])
    return train,test

def split_x_and_y(train,test):
    y_train = train.select("AIA")
    X_train = train.drop("AIA")
    y_test = test.select("AIA")
    X_test = test.drop("AIA")
    return X_train, X_test, y_train, y_test

def resample(base_features,ratio,class_field,base_class):
    pos = base_features.filter(col(class_field)==base_class)
    neg = base_features.filter(col(class_field)!=base_class)
    total_pos = pos.count()
    total_neg = neg.count()
    fraction=float(total_pos*ratio)/float(total_neg)
    sampled = neg.sample(False,fraction)
    return sampled.union(pos)

def create_train_test_and_save():
    train,test = create_train_and_test()
    train = resample(train,1,"AIA",1)
    X_train, X_test, y_train, y_test = split_x_and_y(train,test)
    pd_X_train = X_train.toPandas()
    pd_X_train.to_csv("data/temp/X_train_pd.csv",sep="|")
    pd_X_test = X_test.toPandas()
    pd_X_test.to_csv("data/temp/X_test_pd.csv",sep="|")
    pd_y_test = y_test.toPandas()
    pd_y_test.to_csv("data/temp/y_test_pd.csv",sep="|")
    pd_y_train = y_train.toPandas()
    pd_y_train.to_csv("data/temp/y_train_pd.csv",sep="|")
    
def create_train_test_pd(test_size=.2):
    files = ['data/temp/patents_with_aia_1900_2014.csv','data/temp/patents_with_aia_2014_2018.csv',
            'data/temp/patents_with_aia_2018_2019.csv']
    final_train = pd.DataFrame()
    final_test = pd.DataFrame()
    for fl in files:
        df = pd.read_csv(fl, sep="|")
        train, test = train_test_split(df, test_size=test_size)
        final_test = pd.concat([final_test,test])
        df = train[train['aia']==1]
        df = pd.concat([df,train[train['aia']==0].sample(n=df.shape[0], replace = False, random_state=1)])
        final_train = pd.concat([final_train,df])
    
    return final_train, final_test

def save_train_test(train,test):
    train.to_csv("data/temp/train_pd.csv", sep="|")
    test.to_csv("data/temp/test_pd.csv", sep="|")
    
    
def load_train_test():
    return pd.read_csv("data/temp/X_train_pd.csv", sep="|"),pd.read_csv("data/temp/X_test_pd.csv", sep="|"),\
        pd.read_csv("data/temp/y_train_pd.csv", sep="|"),pd.read_csv("data/temp/y_test_pd.csv", sep="|")