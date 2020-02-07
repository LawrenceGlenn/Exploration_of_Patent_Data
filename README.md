#Natrual Language Processing of US Patent Information
This project is a topic based analysis of the abstracts of patents approved by the US patent office. The potential insights to gain from understanding who is patening what include but are not limited to industry trends, opposition research, and possible converging technologies.
The patent office has over 7 million patents as of the writing of this document with categories ranging over all currently patented techologies and methodologies. This is obviously too much for any person to absorb so by leveraging the power of NLP we can extract insights from this mass of data.

#Data Collection

#EDA
Starting with a focus on the trends of types of patents granted over time we can esially visualize this.

![alt text](/img/count_sections_per_year.jpg "")

Here we can see one category has massivly surpassed all others, but "electricity" is not very informative topic category. To explore this in greater detail text analysis can be performed on the patents within the category labled 'H' or "electricitiy."

#Word Grouping
To determine what is going on in a topic unsupervised learning is performed on a slice of the dataset containing only the "H" section. It might be most instructive to start anaylizing trends in the most recent year to see if there are any interesting upsets or changes in the topics over time.
Performing NLP on the data requires several steps
##removal of stop words
such as 'and', 'the', 'is'
##lemmatization or stemmitization
truncating words into thier roots to make the topic of a document more clear ('runs' 'running' 'run', becomes just 'run')
##tokenization
splitting documents into a list of words ("she has a red dog" becomes ['she', 'has', 'a', 'red', 'dog'])
##TFIDF
create a matrix that represents the freuency of each word as it appears in the total corpus

all of the above steps are performed on the abstracts of our patents. By focusing on the abstracts a maximum amount of information can be extracted about the purpose of a patent with a minimum of words.

#NMF
Non negative matrix factorization is performed on the grouped words and the results plotted for varying numbers of groups to determine what is the best number of topic splits. The results compared via reconstruction error and jaccard similarity.
![alt text](/img/reconstruction_err.jpg "")
![alt text](/img/jiccard_similarity.jpg "")
once a number of topics is selected the model with that many topics can be analyzed for what it represents

#visualization
using pyLDAvis we can more easilly visualize what words are in each topic and how much overlap

Afterwords labels can be placed on each topic and their progression plotted over 2019

#moving forward
Neural networks could improve our clustering abilities, and the use of AWS would allow os to annalyize more data at once.