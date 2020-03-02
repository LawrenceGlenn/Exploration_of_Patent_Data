from src.Import_Patent_Data import  * 
from sklearn.model_selection import GridSearchCV
from src.Pipeline_cleaner import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc

class PatentTrialPredicter:
    
    def __init__(self, train=None, test=None):
        if ((train is None) | (test is None)):
            self.train, self.test = create_train_test_pd()
            self.train, self.test = clean(self.train,self.test)
        else:
            self.train, self.test = train, test
        
    def create_X_y(self):
        self.X_train = self.train.copy()
        self.y_train = self.X_train.pop('aia')
        self.X_test = self.test.copy()
        self.y_test = self.X_test.pop('aia')

            
    def Grid_Search(self,model, params={}):
        grid = GridSearchCV(estimator=model, param_grid=params, scoring = 'recall')
        grid.fit(self.X_train, self.y_train)
        self.grid = grid
        
    def fit_model(self, model, params):
        self.model = model(**params)
        self.model.fit(self.X_train,self.y_train)
        
    def plot_precision_recall(self, ax, label):
        probs = self.model.predict_proba(self.X_test)
        probs = probs[:,1]
        yhat = self.model.predict(self.X_test)
        
        pres, rec, thres = precision_recall_curve(self.y_test, probs)
        self.recall = rec
        self.precision = pres
        # plot the precision-recall curves
        no_skill = len(self.y_test[self.y_test==1]) / len(self.y_test)
        ax.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        ax.plot(rec, pres, label=label)
        # axis labels
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        # show the legend
        ax.legend()