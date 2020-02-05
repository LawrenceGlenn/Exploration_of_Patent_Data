from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

def plot_nmf_tfidf(ax, matrix, num):

	error = [fit_nmf_tfidf(matrix, i).reconstruction_err_ for i in range(1,num+1)]
	ax.plot(range(1,num+1), error)
	ax.set_xticks(range(1, num+1))
	ax.set_xlabel('r')
	ax.set_ylabel('Reconstruction Error')


def fit_nmf_tfidf(matrix, r):
    nmf = NMF(n_components=r)
    nmf.fit(matrix)
    W = nmf.transform(matrix)
    H = nmf.components_
    return nmf