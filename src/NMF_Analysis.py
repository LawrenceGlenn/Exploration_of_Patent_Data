from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_nmf_tfidf(ax, matrix, num, diff=""):

	error, label = compute_error(matrix,num,diff)
	ax.plot(range(1,num+1), error, lw=3)
	ax.set_xticks(range(1, num+1))
	ax.set_xlabel('number of topics',fontsize=20)
	ax.set_ylabel(label,fontsize=20)

def compute_error(matrix,num, diff=""):
	if diff == "jaccard":
		return [compute_jaccard_similarity(matrix,i) for i in range(1,num+1)], 'Jaccard Similarity'
	else:
		return [fit_nmf_tfidf(matrix,i).reconstruction_err_ for i in range(1,num+1)], 'Reconstruction Error'

def compute_jaccard_similarity(matrix,i ):
	nmf = fit_nmf_tfidf(matrix,i)
	H = nmf.components_
	H = pd.DataFrame(H,columns=matrix.columns)

	output = []
	for i in range(len(H)-1):
		x= H[H!=0.0].iloc[i]
		x = x[~np.isnan(x)]
		x=set(x.index)
		y= H[H!=0.0].iloc[i+1]
		y = y[~np.isnan(y)]
		y=set(y.index)
		intersection = set(x).intersection(set(y))
		union = set(x).union(set(y))
		output.append(len(intersection)/len(union))
	return np.mean(output)


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