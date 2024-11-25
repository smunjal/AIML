
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import roc_curve, precision_score, recall_score, accuracy_score, make_scorer, confusion_matrix, f1_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression 
from sklearn.feature_selection import SequentialFeatureSelector,SelectFromModel,RFE, RFECV
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, StratifiedKFold, ShuffleSplit, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import make_column_transformer, TransformedTargetRegressor, make_column_selector, ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SequentialFeatureSelector,SelectFromModel
from sklearn.metrics import mean_squared_error, get_scorer_names,roc_auc_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier 

import scikitplot as skplt


class Config:
    def __init__(self):
        self.globalloglevel = 1
        self.scoring_metric='accuracy'
        self.prediction = False
        self.images_path = "./images/"
        self.palette = ["#fee090", "#fdae61", "#4575b4", "#313695", "#e0f3f8", "#abd9e9", "#d73027", "#a50026"]
        if not os.path.isdir(self.images_path):
            os.mkdir("./images") 
        self.figcounter = 0
        self.train_scores=[]
        self.test_scores=[]
        self.train_mse=[]
        self.test_mse=[]
        self.max_depths = []
        self.mean_fit_times = []
        self.grid_names = []
        self.best_params_arr = []
        self.title_arr=[]
        self.tpr_arr=[]
        self.fpr_arr=[]
        self.dummy_params={}
        self.knn_params={}
        self.lgr_params={}
        self.svc_params={}
        self.tree_params={}



    def getFigTitle(self, title):
        self.figcounter = self.figcounter + 1
        return f'Fig{ self.figcounter } : {title}', self.figcounter

    def getImageDir(self):
        return self.images_path 

    def log(message, loglevel=1):
        if loglevel == self.loballoglevel:
            print(message)
            
    def savefig(self, fig, title):
        fig.savefig(self.images_path + f'{title}.png')
#        fig.savefig(images_path+title)
        fig.show()
        
    def init_globals(self):
        self.train_scores=[]
        self.train_mse=[]
        self.test_scores=[]
        self.test_mse=[]
        self.max_depths = []
        self.mean_fit_times = []
        self.grid_names = []
        self.best_params_arr = []
        self.title_arr=[]
        self.tpr_arr=[]
        self.fpr_arr=[]
        
    def getTitleArr(self):
        return self.title_arr
    def getTprArr(self):
        return self.tpr_arr
    def getFprArr(self):
        return self.fpr_arr
    
    def addConfusionMatrixData(self, title, tpr, fpr):
        self.title_arr.append(title)
        self.tpr_arr.append(tpr)
        self.fpr_arr.append(fpr)
    
    def addModelStats(self, train_scores, train_mse, test_scores, test_mse, mean_fit_times, grid_names, best_params_arr):
        self.train_scores.append(train_scores)
        self.train_mse.append(train_mse)
        self.test_scores.append(test_scores)
        self.test_mse.append(test_mse)
    #    self.max_depths.append(max_depths)
        self.mean_fit_times.append(mean_fit_times)
        self.grid_names.append(grid_names)
        self.best_params_arr.append(best_params_arr)
        
    def get_model_scores(self):
        return self.train_scores, self.test_scores, self.max_depths, self.mean_fit_times, self.grid_names, self.best_params_arr, self.title_arr, self.tpr_arr, self.fpr_arr


    def getDefaultHyperParams(self):
        return {self.dummy_params, self.knn_params, self.lgr_params, self.tree_params, self.svc_params}
    
    def getTypeDict(self):
        return   {1 : 'Dog', 2 : 'Cat'} 
    def getMaturityDict(self):
        return   {1 : 'Small', 2 : 'Medium', 3 : 'Large', 4 : 'Extra Large', 0 : 'Not Specified'}
    def getGenderDict(self):
        return {1 : 'Male', 2 : 'Female', 3 : 'Mixed'} 
    def getFurLenDict(self):
        return {1 : 'Short', 2 : 'Medium', 3 : 'Long', 0 : 'Not Specified'}
    def getVaccinatedDict(self):
        return {1 : 'Yes', 2 : 'No', 3 : 'Not Sure'}
    def getYesNoDict(self):
        return {1 : 'Yes', 0 : 'No'}
    def getHealthDict(self):
        return {1 : 'Healthy', 2 : 'Minor Injury', 3 : 'Serious Injury', 0 : 'Not Specified'}
    def getAdoptionSpeed(self):
        return {0 : '1st Day', 1 : '1st Week', 2 : '1st Month', 3 : '2nd & 3rd month', 4 : 'Not Adopted'}
    def setScoringMetric(self, metric):
        self.scoring_metric = metric            
    def getScoringMetric(self):
        return self.scoring_metric
        


class DatasetHolder:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def update(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    def get(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

'''
Functions to generate visualization
'''
def isPureBreed(type, breed1, breed2) -> int:
    purebred = np.where(((breed1 > 0) & (breed2 == 0)), 1, 0)
    if((purebred == 1) & (type == 1) & (breed1 == 307)):
        purebred = 0
    return int(purebred)


def map_series_to_category(series1, series2, mapper1, mapper2, dependentFeature, dependencymapper):
    tempdf = pd.DataFrame()
    tempdf[series1.name] = series1.map(mapper1)
    tempdf[series2.name] = series2.map(mapper2)
    tempdf[dependentFeature.name] = dependentFeature.map(dependencymapper)
    return tempdf    

def feature_null_percent_with_cutoff(df, cutoff_percent):
    features = []
    for col in df.columns:
        if df[col].isnull().sum()/df.shape[0] * 100 > cutoff_percent:
            features.append(col)
    return features

def feature_null_percentage_in_data(df):
    print(round(df.isnull().sum()/df.shape[0] * 100,2))

def cleaned_data_percent(df):
    cleaned_df = df.dropna()
    print(((df.shape[0] - cleaned_df.shape[0])/df.shape[0])* 100)

def generate_qualitative_plots(data, config, feature, targetfeature, huefeature, feature_desc):

    fig, axs = plt.subplots(2, 2, figsize=(18,12))
    ax = sns.histplot(data=data, x=feature.name, kde=True, ax=axs[0,0])
    axs[0,0].set_title(f"Histogram Plot for : {feature_desc}")
    axs[0,0].grid(visible=True)
   
    mean = np.mean(feature)
    median = np.median(feature)
    stddev = np.std(feature)

    ax.legend([f'Mean: {mean:.2f}', f'Median: {median:.2f}', f'Std Dev: {stddev:.2f}'])
    #ax.axline(mean, color='r', linestyle='--', label='Mean')
    #ax.axline(median, color='g', linestyle='-', label='Median')
    #ax.axline(stddev, color='g', linestyle='-', label='StdDev')
    

    sns.scatterplot(x=feature.name, y=targetfeature.name, hue=huefeature.name, data=data, ax=axs[0,1], alpha=0.8, edgecolors='none')
#    axs[1,1].set_legend(title = 'Target')
    axs[0,1].tick_params(axis='x', rotation=30)
    axs[0,1].set_title(f"Scatter Plot for : {feature_desc} vs {targetfeature.name}")
#    axs[1,1].grid(visible=True)


    corr_data = round(pd.crosstab(targetfeature.sort_values(), feature, normalize=True),2)
    mask = np.triu(np.ones_like(corr_data, dtype=bool))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask
    sns.heatmap(corr_data, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=axs[1,1])    
    # sns.heatmap(corr_data, annot=True, cmap='RdYlGn', ax=axs[1,1])
    axs[1,1].set_title(f'Correlation between {feature_desc} and {targetfeature.name}')
    

    sns.violinplot(x=targetfeature.name, y=feature.name, data=data, ax=axs[1,0], )
    axs[1,0].tick_params(axis='x', rotation=30)
    axs[1,0].set_title(f"Violin Plot for : {feature_desc} vs {targetfeature.name}")
 #   axs[1,0].grid(visible=True)



    title, figcounter = config.getFigTitle(f'Feature {feature_desc} Breakdown')
    fig.suptitle(title)
    config.savefig(fig, title) 
#    plt.savefig(images_path+title)
#    plt.show()

def generate_categorical_plots(data, config, feature, targetfeature, huefeature, feature_desc, show_count=-1, plot_all=True):
    if(plot_all):
        fig, axs = plt.subplots(2, 2, figsize=(18,12))
    #    fig, axs = plt.subplots(1, 2, figsize=(18,6))
        if(show_count > 0):
            sns.countplot(data=data, x=feature.name, hue=targetfeature.name, ax=axs[0,0], palette="rainbow", order = data[feature.name].value_counts().iloc[:show_count].index)
            axs[0,0].set_title(f"Count Plot for Top {show_count} categories in : {feature_desc}")
        else:
            sns.countplot(data=data, x=feature.name, hue=targetfeature.name, ax=axs[0,0], palette="rainbow")
            axs[0,0].set_title(f"Count Plot for categories in : {feature_desc}")
        
        
        axs[0,0].grid(visible=True)
        axs[0,0].set_xlabel(feature_desc)
        axs[0,0].tick_params(axis='x', rotation=45)
    #    axs[0,0].set_ylabel("Counts")

        if(show_count > 0):
            pie_counts = data[feature.name].value_counts().iloc[:show_count]
            axs[0,1].set_title(f"Pie Distribution for Top {show_count} categories in : {feature_desc}")
        else:
            pie_counts = feature.value_counts()
            axs[0,1].set_title(f"Pie Distribution for categories in : {feature_desc}")
        axs[0,1].pie(pie_counts, autopct='%1.0f%%')    
        axs[0,1].legend(labels=pie_counts.index)
        
        tempdf = data.groupby(feature.name)[huefeature.name].value_counts(normalize=True).mul(100).round(2).unstack()
        if(show_count > 0):
            tempdf.plot(kind='bar', stacked=True, colormap='tab10', ax=axs[1,0], order = data[feature.name].value_counts().iloc[:show_count].index) 
    #        sns.boxplot(x=feature.name, y=targetfeature.name, hue=huefeature.name, data=data, palette="coolwarm", ax=axs[1,0], order = data[feature.name].value_counts().iloc[:show_count].index)
            axs[1,0].set_title(f"Stacked Bar Plot for Top {show_count} Type wise % in in : {feature_desc}")
        else:
    #        sns.boxplot(x=feature.name, y=targetfeature.name, hue=huefeature.name, data=data, palette="coolwarm", ax=axs[1,0])
    #        axs[1,0].set_title(f"Box Plot for categories in : {feature_desc}")
            tempdf.plot(kind='bar', stacked=True, colormap='tab10', ax=axs[1,0]) 
            axs[1,0].set_title(f"Stacked Bar Plot for Type wise % in : {feature_desc}")

        axs[1,0].tick_params(axis='x', rotation=30)
        axs[1,0].grid(visible=True)

        tempdf2 = data.groupby(huefeature.name)[feature.name].value_counts(normalize=True).mul(100).round(2).unstack()
        if(show_count > 0):
            tempdf2.plot(kind='bar', colormap='viridis' , stacked=True, ax=axs[1,1], order = data[huefeature.name].value_counts().iloc[:show_count].index) 
    #        sns.violinplot(x=feature.name, y=targetfeature.name, hue=huefeature.name, data=data, ax=axs[1,1], order = data[feature.name].value_counts().iloc[:show_count].index)
            axs[1,1].set_title(f"Bar Plot for Top {show_count} pet type distribution in : {feature_desc}")
        else:
    #        sns.violinplot(x=feature.name, y=targetfeature.name, hue=huefeature.name, data=data, ax=axs[1,1])
    #        axs[1,1].set_title(f"Violin Plot for categories distribution in : {feature_desc}")
            tempdf2.plot(kind='bar', colormap='viridis' , stacked=True,  ax=axs[1,1]) 
            axs[1,1].set_title(f"Bar Plot for pet type wise % in : {feature_desc}")
            
        axs[1,1].tick_params(axis='x', rotation=30)
        axs[1,1].grid(visible=True)
    else:
        fig, axs = plt.subplots(1, 2, figsize=(18,6))
        if(show_count > 0):
            sns.countplot(data=data, x=feature.name, hue=targetfeature.name, ax=axs[0], palette="rainbow", order = data[feature.name].value_counts().iloc[:show_count].index)
            axs[0].set_title(f"Count Plot for Top {show_count} categories in : {feature_desc}")
        else:
            sns.countplot(data=data, x=feature.name, hue=targetfeature.name, ax=axs[0], palette="rainbow")
            axs[0].set_title(f"Count Plot for categories in : {feature_desc}")
        axs[0].grid(visible=True)
        axs[0].set_xlabel(feature_desc)
        axs[0].tick_params(axis='x', rotation=45)
        if(show_count > 0):
            pie_counts = data[feature.name].value_counts().iloc[:show_count]
            axs[1].set_title(f"Pie Distribution for Top {show_count} categories in : {feature_desc}")
        else:
            pie_counts = feature.value_counts()
            axs[1].set_title(f"Pie Distribution for categories in : {feature_desc}")
        axs[1].pie(pie_counts, autopct='%1.0f%%')    
        axs[1].legend(labels=pie_counts.index)
    

    title, figcounter = config.getFigTitle(f'Feature {feature_desc} Breakdown')
    fig.suptitle(title)
    config.savefig(fig, title)
 

### Functions for various operations
## PipelineFactory
## Initialize Global
## Convert captured stats into Dataframe
## Print metrics for pipeline/model
##    * Connfusion Matrix
##    * Classification Report
##    * ROC/AUC Curve



def getTranformer(df):
    categorical_features = get_cat_features(df)
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(sparse_output=False)),
        ]
    )

    count_categorical_transformer = Pipeline(
        steps=[
            ("encoder", ce.CountEncoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", count_categorical_transformer, categorical_features),
        ], remainder='passthrough'
    )

    poly_preprocessor = ColumnTransformer(
        transformers=[
            ("cat", count_categorical_transformer, categorical_features),
            ("poly", PolynomialFeatures(include_bias = False), make_column_selector(dtype_include=np.number)),
        ], remainder='passthrough')
    return preprocessor, poly_preprocessor
    

def logistic_pipeline_factory(transformer):
    dummy = Pipeline([
        ('transformer', transformer), 
        ('scaler', StandardScaler()), 
        ('classifier', DummyClassifier(random_state=42))
    ])

    knn = Pipeline([
        ('transformer', transformer), 
        ('scaler', StandardScaler()), 
        ('classifier', KNeighborsClassifier())
    ])

    lgr = Pipeline([
        ('transformer', transformer), 
        ('scaler', StandardScaler()), 
        ('classifier', LogisticRegression(max_iter=5000))
    ])

    tree = Pipeline([
        ('transformer', transformer), 
        ('scaler', StandardScaler()), 
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])

    svc = Pipeline([
        ('transformer', transformer), 
        ('scaler', StandardScaler()), 
        ('classifier', SVC(random_state=42))
    ])
    return dummy, knn, lgr, tree, svc

def multi_logistic_pipeline_factory(transformer):
    lgr = Pipeline([
        ('transformer', transformer), 
        ('scaler', StandardScaler()), 
        ('classifier', LogisticRegression(multi_class='ovr', max_iter=5000, random_state=42))
    ])

    multi_lgr = Pipeline([
        ('transformer', transformer), 
        ('scaler', StandardScaler()), 
        ('classifier', LogisticRegression(multi_class='multinomial', max_iter=5000, random_state=42))
    ])

    onevsone = Pipeline([
        ('transformer', transformer), 
        ('scaler', StandardScaler()), 
        ('classifier', OneVsOneClassifier(estimator=LogisticRegression(max_iter=5000, random_state=42)))
    ])

    onevsrest = Pipeline([
        ('transformer', transformer), 
        ('scaler', StandardScaler()), 
        ('classifier', OneVsRestClassifier(estimator=LogisticRegression(max_iter=5000, random_state=42)))
    ])

    knn = Pipeline([
        ('transformer', transformer), 
        ('scaler', StandardScaler()), 
        ('classifier', KNeighborsClassifier())
    ])

    tree = Pipeline([
        ('transformer', transformer), 
        ('scaler', StandardScaler()), 
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])
    
    rnforest = Pipeline([
        ('transformer', transformer), 
        ('scaler', StandardScaler()), 
        ('classifier', RandomForestClassifier(random_state=42))
    ])


    svc = Pipeline([
        ('transformer', transformer), 
        ('scaler', StandardScaler()), 
        ('classifier', SVC(random_state=42))
    ])
    return lgr, multi_lgr, onevsone, onevsrest, knn, tree, rnforest, svc



def regression_pipeline_factory(transformer):
    linear = Pipeline([
        ('transformer',  transformer),
        ('scaler', StandardScaler()),
        ('model', LinearRegression(fit_intercept=False) )
    ])

    ridge = Pipeline([
        ('transformer',  transformer),
        ('scaler', StandardScaler()),
        ('model', Ridge() )
    ])

    lasso = Pipeline([
        ('transformer',  transformer),
        ('scaler', StandardScaler()),
        ('model', Lasso(random_state=rs) )
    ])

    fs = Pipeline([
        ('transformer',  transformer),
        ('scaler', StandardScaler()),
        ('selector', SequentialFeatureSelector(LinearRegression(fit_intercept=False))),
        ('model', LinearRegression() )
    ])

    fs_l = Pipeline([
        ('transformer',  transformer),
        ('scaler', StandardScaler()),
        ('selector', SequentialFeatureSelector(Lasso(random_state=42))),
        ('model', LinearRegression() )
    ])
    
    complex = Pipeline([
        ('transformer',  transformer),
        ('scaler', StandardScaler()),
        ('selector', SequentialFeatureSelector(Lasso(random_state=42))),
        ('model', Ridge(random_state=rs))
    ])
    
    ms = Pipeline([
        ('transformer',  transformer),
        ('scaler', StandardScaler()),
        ('selector', SelectFromModel(Lasso(random_state=rs))),
        ('model', LinearRegression() )
    ])
    return linear, ridge, lasso, fs, complex, ms, fs_l


def perform_test(grid, grid_name, config, dataset):
    print(f'=========== Executing - {grid_name} ================')
    X_train, X_test, y_train, y_test = dataset.get() 
    grid.fit(X_train, y_train)
    train_acc = grid.score(X_train, y_train)
    test_acc = grid.score(X_test, y_test)
    train_mse = round(mean_squared_error(y_train, grid.predict(X_train)), 5)
    test_mse = round(mean_squared_error(y_test, grid.predict(X_test)), 5)
    mean_fit_time = grid.cv_results_.get('mean_fit_time').mean()
    print(f'Train Score={train_acc}, Test Score={test_acc}, Mean_fit_time={mean_fit_time}')
    print(grid.best_params_)
    print('==========================================')
    config.addModelStats(train_acc, train_mse, test_acc, test_mse, mean_fit_time, grid_name, grid.best_params_)
    return grid


def dump_df(config):
#    train_scores, test_scores, mean_fit_times, grid_names, best_params_arr, title_arr, tpr_arr, fpr_arr = config.get_model_scores()
    df = {}
    df['Model'] = config.grid_names
    df['Train Time'] = config.mean_fit_times
    df['Train Accuracy'] = config.train_scores
    df['Test Accuracy'] = config.test_scores
    df['Best Params'] = config.best_params_arr
    return pd.DataFrame.from_dict(df)

def dump_tpr_fpr():
#    train_scores, test_scores, mean_fit_times, grid_names, best_params_arr, title_arr, tpr_arr, fpr_arr = config.get_model_scores()
    df = {}
    df['title'] = config.title_arr
    df['Total Positive Rate'] = config.tpr_arr
    df['False Positive Rate'] = config.fpr_arr
    return pd.DataFrame.from_dict(df)

def show_confusion_matrix(dataset, title, grid):
    X_train, X_test, y_train, y_test = dataset.get() 
    y_preds = grid.predict(X_test)
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_preds)
    disp.figure_.suptitle(f"{title} - Confusion Matrix")

def getColumnNames(dataset, fromName, encoded_names):
    X_train, X_test, y_train, y_test = dataset.get() 
    arr = []
    for i in range(0, fnames.size):
        arr.append(X_train.columns[int(fnames[i][1:])])
    return arr

def dump_hyper_params(hyper_params, best_model_params):
    for hp in hyper_params:
        print(f'{hp}={best_model_params.get(hp)}')
        return best_model_params.get(hp)

def dump_feature_imp(dataset, best_estimator):
    X_train, X_test, y_train, y_test = dataset.get() 
    feature_selected_list = selected_columns_list(X_train.columns, best_estimator.named_steps['selector'].get_support())
    print(feature_selected_list)
    coefs_arr = []
    coefs_arr.append(best_estimator.named_steps['model'].coef_)
    pd.DataFrame(coefs_arr, columns=feature_selected_list).head()
#    dump_feature_imp(dataset, best_estimator)

def show_logistic_model_stats(dataset, grid, title, config):
    X_train, X_test, y_train, y_test = dataset.get() 
    y_preds = grid.predict(X_test)
    print_classification_report(grid, title, y_preds, y_test)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].set_title(f"Confusion Matrix : {title}")
    ax[1].set_title(f'ROC Curve : {title}')
    ax[1].grid(visible=True)
    conf_display = show_confusion_matrix(dataset, grid, title, ax[0])
    roc_display = show_roc_curve(dataset, grid, title, ax[1])
    plt.show()
#    print_fpr_tpr(title, y_test, y_preds)

#    show_precision_recall_curve(dataset, grid, title)

def print_fpr_tpr(title, y_true, y_preds, config):
    cm = confusion_matrix(y_true, y_preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = round((fp / (fp + tn) ) * 100, 2) 
    tpr = round((tp / (tp + fn) ) * 100, 2)
    print(f'FPR={fpr}, TPR={tpr}') 
    config.addConfusionMatrixData(title, tpr, fpr)

def show_confusion_matrix(dataset, grid, title,  axs=None):
    X_train, X_test, y_train, y_test = dataset.get() 
    y_preds = grid.predict(X_test)
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_preds, ax=axs)
#    disp = ConfusionMatrixDisplay.from_estimator(grid.best_estimator_, X_test, y_test, ax=axs)
    return disp

def print_classification_report(grid, title, y_preds, y_test):
    print(f"Classification Report - {title}")
    print(classification_report(y_test, y_preds))


def show_roc_curve(dataset, grid, title, axs=None):
    X_train, X_test, y_train, y_test = dataset.get() 
#    disp = RocCurveDisplay.from_estimator(grid.best_estimator_, X_test, y_test,ax=axs);
    y_probas = grid.predict_proba(X_test)
    disp = skplt.metrics.plot_roc(y_test, y_probas, ax=axs);
    return disp

def show_precision_recall_curve(dataset, grid, title,axs=None):
    X_train, X_test, y_train, y_test = dataset.get() 
    disp = PrecisionRecallDisplay.from_estimator(grid.best_estimator_, X_test, y_test, ax=axs)
    disp.figure_.suptitle(f"Precision Recall Curve - {title} ")
    return disp

def find_best_pipe(best_model, lgr, mlgr, ovo, ovr, knn, tree, rntree, svc):

    if best_model == 'Multi-Logistic':
        return mlgr
    elif best_model == 'Logistic Regression':
        return lgr
    elif best_model == 'OneVsOne':
        return ovo
    elif best_model == 'OneVsRest':
        return ovr
    elif best_model == 'KNN':
        return knn
    elif best_model == 'DecisionTree':
        return tree
    elif best_model == 'RandomForest':
        return rntree
    else:
        return svc


def cust_bar_plot(df,title,config, imgName):
    new_title, figcounter = config.getFigTitle(title)
    ax = df.round(2).plot(kind='bar',rot=True, title=new_title, grid=True, figsize=(10,6))
    for container in ax.containers:
        ax.bar_label(container)
#    ax.get_figure().savefig(images_path + imgName)
    config.savefig(ax.get_figure(), new_title)
    return figcounter

'''
def get_custom_scorer(scoring_metric):
    if scoring_metric == 'accuracy':
        return make_scorer(accuracy_score, greater_is_better=True,  pos_label=1)
    elif scoring_metric == 'precision':
        return make_scorer(precision_score, greater_is_better=True,  pos_label=1)
    elif scoring_metric == 'recall':
        return make_scorer(recall_score, greater_is_better=True,  pos_label=1)
    elif scoring_metric == 'f1':
        return make_scorer(f1_score, greater_is_better=True,  pos_label=1)
    elif scoring_metric == 'roc_auc':
        return make_scorer(roc_auc_score, greater_is_better=True)
    else:
        return make_scorer(accuracy_score, greater_is_better=True,  pos_label=1)
'''
def get_custom_scorer(scoring_metric, average='binary'):
    if scoring_metric == 'accuracy':
        return make_scorer(accuracy_score, greater_is_better=True)
    elif scoring_metric == 'precision':
        return make_scorer(precision_score, average=average, greater_is_better=True)
    elif scoring_metric == 'recall':
        return make_scorer(recall_score, average=average, greater_is_better=True)
    elif scoring_metric == 'f1':
        return make_scorer(f1_score, average=average, greater_is_better=True)
    elif scoring_metric == 'roc_auc':
        return make_scorer(roc_auc_score, average=average, greater_is_better=True)
    else:
        return make_scorer(accuracy_score, greater_is_better=True)

def select_top_classifier(df):
    best_model = df.sort_values('Test Accuracy', ascending=False).index[0]
    return best_model



# Create a custom transformer for stemming
class StemmingTransformer:
    def __init__(self):
        self.stemmer = PorterStemmer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [' '.join([self.stemmer.stem(word) for word in word_tokenize(text)]) for text in X]

    def fit_transform(self, X, y=None):
        self = self.fit(X, y)
        return self.transform(X)

# Create a custom transformer for Lemmatizing
class LemmatizingTransformer:
    def __init__(self):
        self.lemma = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [' '.join([self.lemma.lemmatize(word) for word in word_tokenize(text)]) for text in X]

    def fit_transform(self, X, y=None):
        self = self.fit(X, y)
        return self.transform(X)    


def pipeline_factory_vector(vectorizer):
    lgr = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', LogisticRegression(max_iter=10000))
    ])
    tree = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', DecisionTreeClassifier())
    ])
    bayes = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', MultinomialNB())
    ])
    return lgr, tree, bayes

### Commmon Functions for Data Processing

def count_encoder(org_df, categorical_features):
    encoder = ce.CountEncoder()
    encoded_features = encoder.fit_transform(org_df[categorical_features])
    endoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())
    used_cars_df_encoded = org_df.drop(columns=categorical_features).merge(endoded_df, how='inner', left_index=True, right_index=True).reset_index()
    used_cars_df_encoded.drop(columns=['index'], inplace=True)
    used_cars_df_encoded = pd.DataFrame(StandardScaler().fit_transform(used_cars_df_encoded), columns=used_cars_df_encoded.columns)
    return used_cars_df_encoded

def run_price_correlation(df):
    return df.corrwith(df["price"]).sort_values(ascending=False)

def onehot_encoder(org_df, categorical_features):
    encoder = OneHotEncoder(sparse_output=False)
    encoded_features = encoder.fit_transform(org_df[categorical_features])
    endoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())
    used_cars_df_encoded = org_df.merge(endoded_df, how='inner', left_index=True, right_index=True).reset_index()
    print(used_cars_df_encoded.shape)
    used_cars_df_encoded.drop(columns=['index'], inplace=True)
    used_cars_df_encoded.drop(columns=categorical_features, inplace=True)
    used_cars_df_encoded = pd.DataFrame(StandardScaler().fit_transform(used_cars_df_encoded), columns=used_cars_df_encoded.columns)
    return used_cars_df_encoded    

def getX_Y(df):
    X = df.drop('price', axis=1)
    y = df['price']
    return X,y

def get_cat_features(df):
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns
#    print('Numerical Features = ', numerical_features)
#    print('Cateorical Features = ', categorical_features)
    return categorical_features
    
def convert_cat_to_codes(df):
    new_df = df.copy()
    for col_name in new_df.columns:
        if(new_df[col_name].dtype == 'object'):
            new_df[col_name]= new_df[col_name].astype('category')
            new_df[col_name] = new_df[col_name].cat.codes

    return new_df

def variance_comp_count(arr_var, ratio):
    i = 0
    for cumratio in arr_var:
        print(f'{ratio}, {cumratio}')
        if(ratio < int(cumratio)):
            return pca_names
        else:
            pca_names.append(f'pca{i}')
            i = i+1
            return pca_names



def apply_various_cv(pipe, X, y, scorer,current_min_score):
    kfold = KFold(n_splits=5)
    skfold = StratifiedKFold(n_splits=5, shuffle=True)
    ss = ShuffleSplit(n_splits=20, train_size=.4, test_size=.3)

    print('..Running KFold CV')
    kfold_score = cross_val_score(pipe, X, y, cv=kfold)
    kfold_min_score = kfold_score.max()
#    print("KFold: ", kfold_score, kfold_min_score)
    if(kfold_min_score > current_min_score):
        current_min_score = kfold_min_score
        scorer = 'kf'
    print('..Running StratifiedKFold CV')
    strat_kfold_score = cross_val_score(pipe, X, y, cv=skfold) 
    strat_min = strat_kfold_score.max()
#    print("StratifiedKFold:", strat_kfold_score, strat_min)
    if(strat_min > current_min_score):
        current_min_score = strat_min
        scorer = 'stratkf'
    print('..Running ShuffleSplit CV')
    shuffle_split_score = cross_val_score(pipe, X, y, cv=ss)
    shuffle_split_min = shuffle_split_score.max()
#    print("ShuffleSplit: ", shuffle_split_score,shuffle_split_min )
    if(shuffle_split_min > current_min_score):
        current_min_score = shuffle_split_score.min()
        scorer = 'ss'
    return scorer, current_min_score


def dump_coefs_df(model_coefs):
    return pd.DataFrame(model_coefs, columns=X_train.columns, index=models)



def pipeline_proces_and_holdout(pipe, model_name):
    print(f'==================== RUNNING {model_name}=================================')
    pipe.fit(X_train, y_train)
    train_mse = round(mean_squared_error(y_train, pipe.predict(X_train)), 5)
    train_mses.append(train_mse)
    test_mse = round(mean_squared_error(y_test, pipe.predict(X_test)), 5)
    test_mses.append(test_mse)
    train_score = pipe.score(X_train, y_train)
    test_score = pipe.score(X_test, y_test)
#    train_scorer, train_score = apply_various_cv(pipe, X_train, y_train, 'holdout', train_score)
    cv_test_scorer, test_score = apply_various_cv(pipe, X_test, y_test, 'ho', test_score)
    train_scores.append(train_score)
    test_scores.append(test_score)
    scorer.append(cv_test_scorer)
    models.append(f'{model_name}-{cv_test_scorer}')
    result = {'model' :f'{model_name}-{cv_test_scorer}', 'train_mse' : train_mse, 'test_mse' : test_mse, 
              'train_score' :train_score, 'test_score' :test_score, 'scorer' : cv_test_scorer}
    print(result)
    results.append(result)
    print(pipe.named_steps['model'].coef_)
    model_coefs.append(pipe.named_steps['model'].coef_)
    print('==================== DONE =================================================')


def getColumnNames(fromName, encoded_names):
    arr = []
    for i in range(0, fnames.size):
        arr.append(X_train.columns[int(fnames[i][1:])])
    return arr

def dump_hyper_params(hyper_params, best_model_params):
    for hp in hyper_params:
        print(f'{hp}={best_model_params.get(hp)}')
        return best_model_params.get(hp)

def selected_columns_list(cols_arr ,selected_list):
    selected_columns = []
    for i in  range(len(cols_arr)):
        if(selected_list[i]):
            selected_columns.append(cols_arr[i])
    return selected_columns

def dump_feature_imp(dataset, pipeline):
    X_train, X_test, y_train, y_test = dataset.get()
    r = permutation_importance(pipeline, X_test, y_test, random_state=42, n_repeats=30, scoring='r2')
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{X.columns[i]:<18}  "
            f"{r.importances_mean[i]:.3f} "
            f" +/- {r.importances_std[i]:.3f}")


def feature_coef_plots(config, plot_coefs_df):
    # Plot coef
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.barplot(x='feature',
                y='coefficents',
                data=plot_coefs_df, palette=np.array(plot_coefs_df['colors']))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    title, figcounter = config.getFigTitle(figcounter,'Features vs Coeff plot')
    ax.set_title(title)
    ax.set_ylabel('Coefficent Values')
    ax.set_xlabel('Feature Name')
    config.savefig(fig, title)
 
