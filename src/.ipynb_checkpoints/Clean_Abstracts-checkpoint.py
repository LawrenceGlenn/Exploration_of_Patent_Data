#conda install nltk
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_treebank_pos_tagger')
from nltk.corpus import stopwords

stopwords_ = set(stopwords.words('english'))

def clean_and_tokenize_abstract(df):
    #set abstract to lower case
    df = df.withColumn('abstract', lower(col('abstract')).alias('abstract'))
    #remove all non letters
    df = df.withColumn('abstract',regexp_replace(col('abstract'), '\W+', " "))
    # Tokenize text
    tokenizer = Tokenizer(inputCol='abstract', outputCol='abstract_tokenized')
    df = tokenizer.transform(df)
    # Remove stop words
    remover = StopWordsRemover(inputCol='abstract_tokenized', outputCol='abstract_cleaned')
    df = remover.transform(df)
    return df.select('abstract_cleaned')

def patent_data_by_year_and_section(df, year, section_id):
    return df.filter(df['date']==year).filter(df['section_id']==section)

    