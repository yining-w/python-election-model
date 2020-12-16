# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 09:48:38 2020

@author: yinin
"""
#data cite: https://apps.urban.org/features/state-economic-monitor/
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB


def merge_csv(urban_file, year_one, year_two): ##merge state level file from urban database with president file
    path = os.getcwd()
    files = os.listdir(path)
    files = [x for x in files if '.csv' in x]
    
    df = pd.DataFrame()
    
    for each_file in files: #parse urban institute file
        if urban_file in each_file:
            dfTemp = pd.read_csv(each_file,
                                 engine = 'python',
                                 skiprows = 2,
                                 skipfooter = 2,
                                 sep=r',(?!\s|\Z)')
            dfTemp = dfTemp.rename(columns={"State": "state", 'Year': 'year','(D05) Change In Debt During Yr': 'debt_change' })
            dfTemp = dfTemp.loc[(dfTemp['state'] != 'United States')]
            dfTemp = dfTemp.replace({'\$':''}, regex = True)
            dfTemp['debt_change'] = dfTemp['debt_change'].replace({'\0': '', ' ': ''}, regex=True).astype(float)
            dfTemp[["year"]] = dfTemp[["year"]].apply(pd.to_numeric)
            
        elif  '1976-2016-president' in each_file: #parse candidate dataset
            dfTempTwo = pd.read_csv(each_file, engine = 'python')
            dfTempTwo = dfTempTwo.loc[(dfTempTwo['year'] == year_one) | (dfTempTwo['year'] == year_two)]
            dfTempTwo["year"] = dfTempTwo[["year"]].apply(pd.to_numeric)
            dfTempTwo['proportion'] = dfTempTwo['candidatevotes']/dfTempTwo['totalvotes']
        ##only include republican or democrats
    df = pd.merge(dfTempTwo, dfTemp, on = ['state', 'year'])
    #dummy for democrat winning
    df = df[(df.party == "democrat")]
    df['dem_win'] = (df['party'] == 'democrat') & (df['proportion'] > .5)
    df['dem_win'] = df['dem_win'].astype(int)
    df = df.drop(['totalvotes', 'notes', 'writein','state_po', 'state_fips', 'state_cen', 'state_ic', 'office', 'candidate', 'candidatevotes', 'version'], axis=1)
    return df 
         
  
df = merge_csv('results', 2012, 2016)
df.head()
    
## Your training data will be the first presidential election year, and your testing data will be the second.
def divide_samples(df, year_one, year_two):
    X = df.drop(['proportion', 'dem_win' ,'state', 'party', 'proportion'], axis =1)
    X = X.apply(pd.to_numeric)
    Y = df[['dem_win', 'year']]

    X_train = X.loc[X['year'] == year_one]
    X_train = X_train.drop(['year'], axis = 1)
    X_train = preprocessing.scale(X_train)
    #normalize X variable: https://scikit-learn.org/stable/modules/preprocessing.html
    Y_train = Y.loc[Y['year'] == year_one]
    Y_train = Y_train[['dem_win']].values.ravel()
    #cite: https://stackoverflow.com/questions/20868664/should-a-pandas-dataframe-column-be-converted-in-some-way-before-passing-it-to-a

    X_test = X.loc[X['year'] == year_two]
    X_test = X_test.drop(['year'], axis = 1)
    X_test = preprocessing.scale(X_test)
    Y_test = Y.loc[Y['year'] == year_two].reset_index() 
    Y_test = Y_test[['dem_win']].values.ravel()
    
    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test, = divideSamples(df, 2012, 2016)
     
#Use a test harness to assess which model to use

def test_harness():
    #cite: ML Lecture 1
    models = [('Dec Tree', DecisionTreeClassifier()), 
              ('Lin Disc', LinearDiscriminantAnalysis()),  ##different forms of predictors
              ('SVC', SVC(gamma='auto')),
              ('GaussianNB', GaussianNB()),
              ('LogisticRegression', LogisticRegression())]
    results = []
    
    for name, model in models:
        kf = StratifiedKFold(n_splits = 10) #splits data to ten groups, runs tests across the groups
        res = cross_val_score(model, X_train, Y_train, cv=kf, scoring='accuracy')
        res_mean = round(res.mean(), 4)
        res_std  = round(res.std(), 4)
        results.append((name, res_mean, res_std))
        print('Model:', name, 'Mean Accuracy Score:', res_mean, 'Std:', res_std)

test_harness()
##Results show Logistic Regression with highest accuracy score and lowest standard deviation.

#Fit a supervised ML model (classification) to the data in that way that lets you make predictions,
# then compare the predictions to the actual outcome.
def predict_model(df, model_name, year_two):
    model = model_name
    model.fit(X_train, Y_train)
    predict = model.predict(X_test)

    compare = pd.DataFrame({'predicted': predict, 'observed': list(Y_test)}, columns =['predicted', 'observed']) 
    df = df.loc[df['year'] == year_two] 

    f_column = df["state"].reset_index()
    compare = pd.concat([compare, f_column], axis = 1)
    compare['match'] = (compare['predicted'] == compare['observed'])
    compare['match'] = compare['match'].astype(int)
    return compare
    
comparison = predict_model(df, LogisticRegression(), 2016)
print(comparison)
comparison.match.sum()
##32 out of 52 matches

#End your code with a few lines discussing what you found.

#-- I found that the code accurately predicted only 61% of 2016 results by state, 
#though I did not find a ton of independent variables to measure the results with.

#Did your model do a good job of predicting the winner? 
#-- My model did not do a good job at predicting the winner. Given the heterogenuous 
#qualities across states, it is unrealistic to predict winner based on a few economic/
#indicators. It would have also been helpful to use datasets that looked at
#demographical differences such as race, socioeconomic distribution, and occupations. 
#I am also wary of using different years as validation / testing data, and wonder
#if there is a different method we can divide the samples instead.
#particularly, i feel that the approach doesn't take things that change all units
#across time. In addition, a prediction for an incumbent year vs a new president may
#also create changes in predictions between two time periods.

#What accuracy measure did you use to evaluate the model results, and why?
## I used the accuracy classification score. Since I designated the dependent variable,
## democrat_win, as a binary for 0 if a democratic candidate loses and 1 if they win,
# the set of predicted labels must exactly match the validation data as a measure of accuracy.

#Were you able to reuse any code that you wrote previously, and if so, 
#what worked well or didn't work well?
## I used my previous csv reader code from assignment 1. It generally worked well,
#though I tried to generalize my working directory this time. In addition,
## I had to code data cleaning separately because the dataset I wanted to merge the
#assignment's data set with were formatted differently and had different sets of issues.
