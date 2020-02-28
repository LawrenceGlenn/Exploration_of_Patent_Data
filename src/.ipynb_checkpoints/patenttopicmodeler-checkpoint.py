
from .Clean_Abstracts import *
from .NMF_Analysis import *
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification
import pyLDAvis
import pyLDAvis.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
pyLDAvis.enable_notebook()

class PatentTopicModeler:

	def __init__(self, df):
		self.df = df

	def clean_df(self):
		clean_df = clean_and_abstract_pd(self.df)
		clean_df = lem_abstract_pd(clean_df)
		clean_df = stem_abstract_pd(clean_df)
		self.df = clean_df

	def tfidf(self,col = "abstract_cleaned",ngram1 = 2, ngram2=3):
		corpus = [" ".join(row) for row in self.df[col]]
		self.tfidf_vector = TfidfVectorizer(ngram_range=(ngram1,ngram2),min_df=.01,max_df=.6)
		self.document_tfidf_matrix = self.tfidf_vector.fit_transform(np.array(corpus))

		self.tfidf_matrix = pd.DataFrame(self.document_tfidf_matrix.toarray(), columns= self.tfidf_vector.get_feature_names())

	def plot_tfidf(self,ax,width=12):
		plot_nmf_tfidf(ax[0],self.tfidf_matrix, 12)
		plot_nmf_tfidf(ax[1],self.tfidf_matrix, 12, 'jaccard')

	def fit_nmf(self, comp = 7):
		nmf = fit_nmf_tfidf(self.tfidf_matrix, comp)
		self.nmf = nmf
		self.W = nmf.transform(self.tfidf_matrix)
		self.H = nmf.components_

	def top_words(self,num = 20):
		print(top_words(self.H,self.tfidf_matrix.columns,num))

	def fit_lda(self, comp = 7, rand = 0):
		lda = LatentDirichletAllocation(n_components=comp,random_state=rand)
		lda.fit(self.tfidf_matrix)

		# get topics for some given samples:
		lda.transform(self.tfidf_matrix[-2:])
		self.lda = lda

	def pyldavis(self):
		pyLDAvis.sklearn.prepare(self.lda, self.document_tfidf_matrix, self.tfidf_vector)

	def assign_nmf_topic_id_to_df(self):
		ids = (np.around(x,self.W.shape[0]) for x in W)
		ids_df = pd.DataFrame(ids,index=self.df.index)
		misc = (W.shape[0]+1).to_str()
		ids_df[misc]=0
		ids_df.loc[ids_df.sum(axis=1)==0,misc]=2
		self.df['topic_id'] = ids_df.idxmax(axis=1)

	def plot_topics(self,ax,maxDate = '30000', minDate = '2019'):
		graph_df = self.df.groupby(['date','topic_id']).count()['patent_id'].reset_index().rename(columns={'patent_id':'count'}).pivot(index='date',columns='topic_id',values='count')
		graph_df = graph_df[graph_df.index>=minDate]
		graph_df = graph_df[graph_df.index<=maxDate]
		graph_df.plot(ax=ax, logy=False, lw=5)
		