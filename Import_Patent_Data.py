import pyspark as ps    # for the pyspark suite
import pandas as pd

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
	return df_with_groups


if __name__ == __main__:
	return import_data()