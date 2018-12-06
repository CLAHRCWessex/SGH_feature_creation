import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report,precision_score, recall_score, confusion_matrix,accuracy_score
from sklearn.metrics import precision_recall_curve,roc_curve

from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV

def plot_raw_timeseries_for_basic_vis_check(df,col_list):
    """plots timeseries' of choice so can check visually if any major problems with data"""
    #### calc number of rows
    if (len(col_list)%3) == 0:
        rows = int(len(col_list)/3)
    else:
        rows = (len(col_list)//3) + 1


    fig,ax = plt.subplots(rows,3,figsize=(18,3*rows))

    if len(ax) != 3:
        ax = [item for sublist in ax for item in sublist] # flatten list of axes if more than one rows

    for i in np.arange(len(col_list)):
        df[col_list[i]].plot(ax=ax[i])
        ax[i].set_title(col_list[i])
        ax[i].set_xlabel('')
    return

def find_feature_types(df):
    """takes df and provides a list of numeric,binary,catagorical variables"""
    #### make df which lists the variable types and how many unique value there are for each.
    s1 = df.nunique()
    s2 = df.dtypes
    typesdf = pd.concat([s2,s1], keys=['type','value_count'],axis=1)

    def search_for_string(typesdf,search_string):
        """look in features for name containing a string returns list of column names"""
        year_flag_list = []
        for i in list(typesdf.index):
            #print(i)
            if (search_string in i) == True:
                year_flag_list.append(i)
        return(year_flag_list)

    #### get list of var types
    mask = ((typesdf['type'] == 'int64')|(typesdf['type'] == 'int32')|(typesdf['type'] == 'float64')) & (typesdf['value_count'] >12)
    num_attribs = list(typesdf[mask].index) + search_for_string(typesdf,'YEAR') # final + add all vars with YEAR to list
    mask = ((typesdf['type'] == 'int64')|(typesdf['type'] == 'int32')|(typesdf['type'] == 'float64')) & (typesdf['value_count'] <= 12)& (typesdf['value_count'] > 2)
    cat_attribs = list(typesdf[mask].index)
    # remove all YEAR instances from catagories
    for feature_name in search_for_string(typesdf,'YEAR'):
        if feature_name in cat_attribs: cat_attribs.remove(feature_name)
    # only include features with refs to weekday, month
    only_include_cols = search_for_string(typesdf,'WEEKDAY') + search_for_string(typesdf,'MONTH')
    cat_attribs_filt = []
    for allowed_col in only_include_cols:
        if allowed_col in cat_attribs: cat_attribs_filt.append(allowed_col)

    mask = ((typesdf['type'] == 'int64')|(typesdf['type'] == 'int32')|(typesdf['type'] == 'bool')) & (typesdf['value_count'] == 2)
    bin_attribs = list(typesdf[mask].index)

    #### warn of columns which ahve been ignored
    cols_included = num_attribs + cat_attribs_filt + bin_attribs
    print('Number of cols: ', df.shape[1])
    print('Numerical: ', len(num_attribs))
    print('Catagorical: ', len(cat_attribs_filt))
    print('Binary: ', len(bin_attribs))
    print('Total used: ', len(cols_included))
    print()
    print('WARNING UNUSED COLUMNS:')
    for i in df.drop(cols_included,axis=1).columns:
        print(i, ',',df[i].dtype )

    return(num_attribs,cat_attribs_filt,bin_attribs,typesdf)


from sklearn.base import BaseEstimator, TransformerMixin

class LaggedFeaturesForSelectedVars(BaseEstimator, TransformerMixin):
    """take df and dictionary of which columns to create lagged vas
    dict keys contain column name and values contain list of int (time lags)
    e.g. {'COUNT(EDatt)':[1,3,7,30]}"""

    def __init__(self, col_changes):
        self.col_changes = col_changes
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        for col_name in self.col_changes: #loop by columns (keys in dict)
            print(col_name)
            for i in self.col_changes[col_name]: #loop by value in dict correpsonding to key above
                #print(i)
                X[col_name+'_lag'+str(i)] = X[col_name].shift(i) # shift of each column and saving to new column with subscript
                X[col_name+'_sum'+str(i)] = X[col_name].rolling(i).sum()
                X[col_name+'_mean'+str(i)] = X[col_name].rolling(i).mean()
        return X


class LaggedFeaturesAll(BaseEstimator, TransformerMixin):
    """take df and returns lagged values
    dict: keys contain type of lagged feature (str) and list of lags to apply over (list of int)
    """

    def __init__(self, feature_changes,cols_to_change):
        self.feature_changes = feature_changes
        self.cols_to_change = cols_to_change
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        for col_name in X[self.cols_to_change].columns: #loop by df columns (but only columns included in cols_to_change)
            #print(col_name)
            for lag_type in self.feature_changes.keys():
                if lag_type == 'lag':
                    for i in self.feature_changes[lag_type]:
                        X[col_name+'_lag'+str(i)] = X[col_name].shift(i) # shift of each column and saving to new column with subscript
                if lag_type == 'sum':
                    for i in self.feature_changes[lag_type]:
                        X[col_name+'_sum'+str(i)] = X[col_name].rolling(i).sum()
                if lag_type == 'mean':
                    for i in self.feature_changes[lag_type]:
                        X[col_name+'_mean'+str(i)] = X[col_name].rolling(i).mean()

        return X

class OffsetTargetVar(BaseEstimator, TransformerMixin):
    def __init__(self, col_changes):
        self.col_changes = col_changes
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        for col_name in self.col_changes: #loop by columns (keys in dict)
            #print(col_name)
            for i in self.col_changes[col_name]: #loop by value in dict correpsonding to key above
                #print(i)
                X[col_name+'_lagN'+str(i)] = X[col_name].shift(-i) # shift of each column and saving to new column with subscript
        return

def create_binary_class_label_rollingSTD(df,target_col,period=90,stds=1.2):
    """take df and target column, create new column with labels where label '1' is the extreme cases above the rolling STD.
    print proportion of '1' labels
    produce plot to show flagged points as '1's
    returns df with new columns - info about threshold too"""
    #### get threhold values
    df[target_col + '_rollingMED'] = df[target_col].rolling(period).median()
    df[target_col + '_rollingSTD'] = df[target_col].rolling(period).std()
    df[target_col + '_rollingTHRESH'] = (df[target_col + '_rollingMED'] + (stds*df[target_col + '_rollingSTD']))
    #### make target column (binary)
    df['target_class'] = (df[target_col] > df[target_col + '_rollingTHRESH']).astype(int)

    #### print prop of 1's
    print('Proportion of points classed as state 1: ' + str((df.target_class.value_counts()[1]/df.shape[0]).round(3)))
    #### plot
    ax = plt.subplot()
    df[target_col].plot(ax=ax,figsize=(17,5))
    df[df.target_class == 1][target_col].plot(ax=ax,style='.k')

    return(df)



def make_test_train_splits(df,target_col,target_class_col,target_class_colLAGGED,test_size,valid_size=0):
    """takes sizes of test,train,valid splits.
    creates new dfs for each
    removes target column from dfs.
    prints sizes of each split
    plots timeseries of data."""
    #### make splits
    X_test = df[-test_size:]
    X_valid = df[-(test_size + valid_size):-test_size]
    X_train = df[:-(test_size + valid_size)]
    X_train_valid = df[:-test_size]

    y_test = X_test.pop(target_class_colLAGGED)
    y_valid = X_valid.pop(target_class_colLAGGED)
    y_train = X_train.pop(target_class_colLAGGED)
    y_train_valid = X_train_valid.pop(target_class_colLAGGED)

    #### print sizes of data sets
    print('DATA POINTS:')
    sizes = {'training: ':(df.shape[0] - test_size -  valid_size), 'validation: ':valid_size,'testing: ': test_size}
    for i in sizes:
        print(i,sizes[i])

    #### plot train ,valid,test in timeseries

    fig, ax = plt.subplots(figsize=(18,6))
    #### plot lines
    X_train.plot(x="arr_date", y=target_col, ax=ax, label="train")
    X_test.plot(x="arr_date", y=target_col, ax=ax, label="test")
    if valid_size != 0:
        X_valid.plot(x="arr_date", y=target_col, ax=ax, label="validate")


    df[df[target_class_col] == 1].plot(x='arr_date',y=target_col,ax=ax,style='.k')
    plt.legend(loc='upper left')
    # plt.savefig('images/svm-split.png');

    return(X_test,X_valid,X_train,X_train_valid,y_test,y_valid,y_train,y_train_valid)

# feature_matrix = feature_matrix.T.drop_duplicates().T # this made problems as converted all dtypes to obejcts
def duplicate_columns(frame):
    "finds columns with duplicate values in each row. Returns list of strings."
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            iv = vs.iloc[:,i].tolist()
            for j in range(i+1, lcs):
                jv = vs.iloc[:,j].tolist()
                if iv == jv:
                    dups.append(cs[i])
                    break

    return dups

def DataFrameRebuild(df,X,num_attribs=None,cat_attribs=None,bin_attribs= None):
    """takes original df and transformed numpy-array
    and re-creates df with column names"""
    names = []
    if num_attribs != None:
        names += list(df[num_attribs].columns)

    if cat_attribs != None:
        for i in df[cat_attribs]:
            cats_var_list = list(df[i].value_counts().index)
            for j in cats_var_list:
                names += ([i+'_'+str(j)])

    if bin_attribs != None:
        names += list(df[bin_attribs].columns)

    #### make df from array
    X = pd.DataFrame(data=X,columns=names)
    return(X)

def plot_model_perf(ydata,Xdata,model,label):
    """
    plot:
    recall, prec, vs. threshold
    prec vs. recall
    ROC
    """
    #### get decision function
    if hasattr(model, 'predict_proba'):
        ydata_predDF = model.predict_proba(Xdata)[:,1]
    else:
        ydata_predDF = model.decision_function(Xdata)# warning, some model dont have DF
        print('No predicit_proba, DF used instead')

    #### get threshold + ROC curve data
    precisions, recalls, thresholds = precision_recall_curve(ydata,ydata_predDF)
    fpr,tpr,thresholds_ROC = roc_curve(ydata,ydata_predDF)

    #### create plot
    fig,ax = plt.subplots(1,3,figsize=(15,4))
    fig.suptitle(label)

    #def plot_precision_recall_vs_threshold(precisions, recalls,thresholds):
    ax[0].plot(thresholds,precisions[:-1],'b--',label='Prec')
    ax[0].plot(thresholds,recalls[:-1],'g-',label='Reca')
    ax[0].set_xlabel('Threshold')
    ax[0].set_ylabel('')
    ax[0].legend(frameon=True,loc='center right')
    ax[0].set_ylim([0,1])

    ax[1].plot(recalls[:-1],precisions[:-1],'g-') #,label='Reca')
    ax[1].set_ylabel('Precision')
    ax[1].set_xlabel('Recall')
    ax[1].legend(frameon=True,loc='center right')
    ax[1].set_ylim([0,1])

    ax[2].plot(fpr,tpr)
    ax[2].plot([0,1],[0,1],'k--')
    ax[2].set_xlabel('F positive rate')
    ax[2].set_ylabel('T positve rate')

    #print model perforamnce stats
    y_pred = model.predict(Xdata)
    print('accuracy: ',accuracy_score(ydata,y_pred).round(2))
    print('precision: ', precision_score(ydata,y_pred).round(2))
    print('recall: ', recall_score(ydata,y_pred).round(2))

    return
