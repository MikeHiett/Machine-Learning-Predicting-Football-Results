#this is the prediction engine for the data

import pandas as pd
import xgboost as xgb

def predictResults(completedSeasons, currentSeason, nextWeek):
    
    newWeek = nextWeek.copy()
    
    trainingData = pd.concat([completedSeasons,
                              currentSeason], ignore_index=True)
    
    #remove the first 5 matchweeks from training set and drop unneccessary columns
    trainingData = trainingData[trainingData.MW > 5]
    
    trainingData.drop(['Unnamed: 0','HomeTeam','AwayTeam','Date','FTHG','FTAG','HTF3','ATF3','HTF5','ATF5','HomeTeamLP','AwayTeamLP','MW','DiffLP'],1, inplace=True)
    newWeek.drop(['HomeTeam','AwayTeam','Date','FTHG','FTAG','HTF3','ATF3','HTF5','ATF5','HomeTeamLP','AwayTeamLP','MW','DiffLP','FTR'],1, inplace=True)

    
    #split the data into features and predictor
    X_all = trainingData.drop(['FTR'],1)
    y_all = trainingData['FTR']
    
    #identify the parameters for the classifier
    
    parameters = { 'learning_rate' : [0.1],
                   'objective': 'multi:softprob',
                   'num_class' : [3],
                   'n_estimators' : [40],
                   'max_depth': [3],
                   'min_child_weight': [3],
                   'gamma':[0.4],
                   'subsample' : [0.8],
                   'colsample_bytree' : [0.8],
                   'scale_pos_weight' : [1],
                   'reg_alpha':[1e-5]
                 } 
    
    #initialise the classifier
    print('Initialising the classifier')
    clf = xgb.XGBClassifier(parameters)
    print('Training the classifier using results data')
    clf.fit(X_all, y_all)
    print('Making this weeks predictions')
    predict = clf.predict(newWeek)
        
    return predict
