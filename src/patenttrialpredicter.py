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
        
        
if __name__ == "__main__":
    train = pd.read_csv("sampleData/train_pd.csv",sep="|")
    test = pd.read_csv("sampleData/train_pd.csv",sep="|")
    train,test = tfidf_abstract(train,test)
    
    train,test = clean(train,test)
    train = remove_columns(train,['num_claims','withdrawn', 'Unnamed: 0'])
    test = remove_columns(test,['num_claims','withdrawn','Unnamed: 0'])
    pat_modeler = ptp.PatentTrialPredicter(train,test)
    pat_modeler.create_X_y()

    parameters_gb = {
        'learning_rate': [0.1],
        'max_depth': [80, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [1, 3],
        'min_samples_split': [8, 10],
        'n_estimators': [50, 100, 200]
    }
    model_gb = GradientBoostingClassifier()
    pat_modeler.Grid_Search(model_gb,parameters_gb)
    pat_modeler.grid.best_score_
    pat_modeler.fit_model(GradientBoostingClassifier, pat_modeler.grid.best_params_)
    fig,ax = plt.subplots()
    pat_modeler.plot_precision_recall(ax, "Gradient Boosting")
    
    fig,ax = plt.subplots(figsize = (16,16))
    plot_feature_importance(ax,pat_modeler.X_train.columns.tolist()[1:],pat_modeler.model.feature_importances_.tolist())