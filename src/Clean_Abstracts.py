#conda install nltk
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_treebank_pos_tagger')
from nltk.corpus import stopwords
import pyspark as ps
from pyspark.sql.functions import udf,lower, col, regexp_extract, regexp_replace
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from nltk.stem import WordNetLemmatizer
from pyspark.sql.types import ArrayType, StringType

stopwords_ = set(stopwords.words('english'))

def clean_and_tokenize_abstract(df):
    #set abstract to lower case
    df = df.withColumn('abstract_low', lower(col('abstract')).alias('abstract_low'))
    #remove all non letters
    df = df.withColumn('abstract_remove',regexp_replace(col('abstract_low'), '[\W\d]+', " "))
    # Tokenize text
    df = df.drop('abstract_low')
    tokenizer = Tokenizer(inputCol='abstract_remove', outputCol='abstract_token')
    df = tokenizer.transform(df)
    df = df.drop('abstract_remove')
    # Remove stop words
    remover = StopWordsRemover(inputCol='abstract_token', outputCol='abstract_cleaned')
    df = remover.transform(df)
    df = df.drop('abstract_token')
    return df


def lem_abstract(df):
# Lem text
    lemmer = WordNetLemmatizer()
    lemmer_udf = udf(lambda tokens: [lemmer.lemmatize(token) for token in tokens], ArrayType(StringType()))
    df = df.withColumn("abstract_lemmed", lemmer_udf("abstract_cleaned"))
    df = df.drop('abstract_cleaned')
    return df

def patent_data_by_year_and_section(df, year, section):
    return df.filter(df['date']==year).filter(df['section_id']==section)

def clean_and_abstract_pd(df):
    #set abstract to lower case
    df['abstract_low'] = df['abstract'].str.lower()
    #remove all non letters
    df['abstract_low'] = df['abstract_low'].str.findall(r'([a-z]+)')
    # Remove stop words

    stopwords_ = set(stopwords.words('english'))
    df['abstract_cleaned'] = df['abstract_low'].apply(lambda x: [item for item in x if item not in stopwords_])
    df = df.drop("abstract_low",axis=1)
    return df

def lem_abstract_pd(df):
    word_lem = WordNetLemmatizer()
    df['abstract_lemmed'] = df['abstract_cleaned'].apply(lambda x: [word_lem.lemmatize(y) for y in x])
    df = df.drop('abstract_cleaned',axis=1)
    return df