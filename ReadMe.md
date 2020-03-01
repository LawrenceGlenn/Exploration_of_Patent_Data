# Will your patent cost you?

Every business needs patents. Taking ownership of your creations and creation methods is a nessicary step in future profitability of any sucessful business. We all know that it costs a lot of money to file a patent but costs can continue later in a patents lifetime.

The creation of The Patent Trials and Appeals Board (PTAB) has allowed for easier and cheaper litigation for any claiment who wishes to challenege an exsisting patent. This puts pressure like never beore on companies small and large alike to be certain that the patents they file are not only capable of passing the process of being approved but also to make them iron clad to future challenges.

That is where this program comes in. By creating machine learning models a reliable assessment can be made of any filed patent determining the liklihood it will be called before PTAB.

## The Data: Obtaining, Wrangling and Cleaning

### Obtaining
Publicly available patent data can be obtained at https://www.patentsview.org/download/. The primary data set is labeled as patent as of the writing of this readme. Other datasets can also be downloaded here and their information added as columns to the data set to alter the model. Data experimened with included uspc, wipo and cpc_current.

The PTAB office has an API that can be used to obtain information on patents called before the board. On https://developer.uspto.gov/ptab-api/swagger-ui.html open the get proceedings section, in the field labeled proceedingTypeCategory input AIA and in the field labeled recordTotalQuantity input a number higher than the total number of AIA trials (11227 as of the writing of this readme) and click try it out. This will return all information on all AIA trials and can be copied into a file or database of your choice.

### Wrangling
The data is far too large to anaylize in a pandas dataframe even with the memory available on an AWS instance. The final solution was to read in patent data a piece at a time via chunksize and only a few years with of data at a time, label each row with a 1 or 0 based on their presence or absence in the ptab data and then save the information to seperate files. The ratio of 1 to 0 in this case is approximitly 1000:1, meaning we will need to undersample our data and the resulting dataset will be small enough to handle with available RAM. Functions to read in, seperate, and sample from this data are available in the src folder of this project.

## EDA

## Model Creation and testing

## Conclusions



## Future Directions

LDA analysis of the full text of patents would provide new features that might be instructive to the preditive model.
One class classification could be created and results compared to current model.
LSTM network could be compared to current model.
Investigate method to cobtain size of company information to add as a feature to future models.

TODO
MORE EDA
GRAPHS
CROSS VAL AFTER GRID SEARCH AND DISPLAY OF INFO
CLEANUP
LOGISTIC REGRESSION
KNN