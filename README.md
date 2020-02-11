# Natrual Language Processing of US Patent Information
This project is a topic based analysis of the abstracts of patents approved by the US patent office. The potential insights to gain from understanding who is patening what include but are not limited to industry trends, opposition research, and possible converging technologies.
The patent office has over 7 million patents as of the writing of this document with categories ranging over all currently patented techologies and methodologies. This is obviously too much for any person to absorb so by leveraging the power of NLP we can extract insights from this mass of data.

# Data Collection
All of our data can be obtained from either www.uspto.gov or www.patentsview.org. The focus of this project will be inside the patents.tsv table joined on patent ids with the cpc_current table. Drop all the but the last squence to eliminate duplicate patent ids and remove all columns except patent_id, date, section_id, title, and abstract to leave the pertinent information.

# EDA
Starting with a focus on the trends of types of patents granted over time we can esially visualize this.

![alt text](/img/count_sections_per_year.jpg "")

Here we can see one category has massivly surpassed all others, but "electricity" is not very informative topic category. To explore this in greater detail text analysis can be performed on the patents within the category labled 'H' or "electricitiy."

# Word Grouping
To determine what is going on in a topic unsupervised learning is performed on a slice of the dataset containing only the "H" section. It might be most instructive to start anaylizing trends in the most recent year to see if there are any interesting upsets or changes in the topic recent history.
Performing NLP on the data requires several steps
### removal of stop words
such as 'and', 'the', 'is'
### lemmatization or stemmitization
truncating words into thier roots to make the topic of a document more clear ('runs' 'running' 'run', becomes just 'run')
### tokenization
splitting documents into a list of words ("she has a red dog" becomes ['she', 'has', 'a', 'red', 'dog'])
### TFIDF
create a matrix that represents the frequency of each word as it appears in the total corpus.

All of the above steps are performed on the abstracts of the patents. By focusing on the abstracts a maximum amount of information can be extracted about the purpose of a patent with a minimum of data.

# NMF
Non negative matrix factorization is performed on the grouped words and the results plotted for varying numbers of groups to determine what is the best number of topic splits. The results compared via reconstruction error and jaccard similarity are displayed below.

![alt text](/img/reconstruction_err_section_H.png "")

![alt text](/img/jaccard_err_section_H.png "")

The reconstruction error indicates that no matter how many topics the words are sliced into the amount of diffierentiation gained improves by the same amount.

The jaccard similarity measure how different two sets are. There is an increase in similarity of the topics at 8 so 7 topics was selected but other numbers can be equally valid.

once a number of topics is selected the model with that many topics can be analyzed for what it represents

# Visualization
using pyLDAvis the topics can be more easilly visualized displaying what words are in each topic and how much overlap as well as obtain convenient lists of the top most relavent words.
the html of this display is saved in this project under Ida.html

The topics assigned to each created category are determined subjectivly but for selection are labeled as follows.
1) Networkd Components
2) Semiconductors, Power Supplies
3) Methodologies, Communications
4) Lights, Circuitboard Layers/Construction
5) Complex Constructions (ie. multi part components)
6) Connumications, Wireless
7) Mobile Devices
8) Misc.

Afterwords the labels are placed on each topic their progression can be plotted over 2019

![alt text](/img/electricity_2019_topics.png "")

# Conclusions
It is clear that certain technologies are more popular than others however they follow essentially the same trends inside 2019. This is not a particularly unexpected result as the project was limited to 1 year for hardware limitation reasons. Given enough memory the same process could be repeated over the entire length of the dataset (1976-2019) topics derived and plotted which may provide greater illumination on patent trends over time.

# moving forward
Neural networks could improve clustering abilities, and the use of AWS would allow annalyization of more data at once.