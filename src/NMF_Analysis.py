from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_nmf_tfidf(ax, matrix, num):

	error = [fit_nmf_tfidf(matrix, i).reconstruction_err_ for i in range(1,num+1)]
	ax.plot(range(1,num+1), error)
	ax.set_xticks(range(1, num+1))
	ax.set_xlabel('r')
	ax.set_ylabel('Reconstruction Error')


def fit_nmf_tfidf(matrix, r):
    nmf = NMF(n_components=r)
    nmf.fit(matrix)
    return nmf

def display_W_H(W,H,index,columns,comp):

	# Make interpretable
	W, H = (np.around(x,comp) for x in (W,H))
	W = pd.DataFrame(W,index=index)
	H = pd.DataFrame(H,columns=columns)

	display(W) 
	display(H)

def top_words(H, columns, num):
	H = pd.DataFrame(H,columns=columns)
	for i in range(H.shape[0]):
		print(H.sort_values(by=i, ascending=False,axis=1).iloc[i:i+1,:num])
		
def remove_words_from_df(df,rm):
    df=df.copy()
    for i,row in enumerate(df['abstract_lemmed']):
   # print([x for x in row if x not in rm])
        df['abstract_lemmed'][i]=[x for x in row if x not in rm]
    return df