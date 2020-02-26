
from Clean_Abstracts import *
class PatentTopicModeler:

	def __init__(self, df):
		self.df = df

	def clean_df():
		clean_df = clean_and_abstract_pd(self.df)
		clean_df = lem_abstract_pd(clean_df)
		clean_df = stem_abstract_pd(clean_df)
		self.df = clean_df

	def tfidf(col = "abstract_cleaned",ngram1 = 2, ngram2=3):
		self.tfidf_matrix = tfidf_vector_matrix(self.df,ngram1,ngram2,col)

	def plot_tfidf(ax,width=12):
		plot_nmf_tfidf(ax[0],self.tfidf_matrix, 12)
		plot_nmf_tfidf(ax[1],self.tfidf_matrix, 12, 'jaccard')