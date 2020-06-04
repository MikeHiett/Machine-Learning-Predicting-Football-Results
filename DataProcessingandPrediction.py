#The features being used for the prediction will be:
# - Team Form ( last 5 results )
# - Team form ( last 3 results )
# - Difference in current league points
# - Difference in current goal difference
# - Home Team Home Win Percentage
# - Away Team Away Win Percentage
# - Final Position in Previous Season

#Import dependencies
print('Importing Dependend Modules')
import numpy as np
import pandas as pd
import urllib.request
import xgboost as xgb
#download the most recent version of the results data



print('Downloading match results..')
url = 'https://www.football-data.co.uk/mmz4281/1920/D1.csv'
urllib.request.urlretrieve(url, 'C:/Users/Michael/Desktop/FPLPredictionsWebsite/MachineLearningEPL/Bundesliga/Bundesliga_1920.csv')


#read the data from the CSV into a dataframe

raw1920 = pd.read_csv('C:/Users/Michael/Desktop/FPLPredictionsWebsite/MachineLearningEPL/Bundesliga/Bundesliga_1920.csv')

#pass only the columns that are required
playing_statistics_1920 = raw1920[['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR']]

#get the last matchweek and tag on the next matchweeks fixtures

def get_mw(playing_stat):
    j = 1
    MatchWeek = []
    for i in range(len(playing_stat)):
        MatchWeek.append(j)
        if ((i + 1)% 9) == 0:
            j = j + 1
    playing_stat['MW'] = MatchWeek
    return playing_stat

print('Finding Current Matchweek and next Matchweeks Fixtures')
playing_statistics_1920 = get_mw(playing_statistics_1920)


fixtures = pd.read_csv('C:/Users/Michael/Desktop/FPLPredictionsWebsite/MachineLearningEPL/Bundesliga/Fixtures.csv',encoding = "ISO-8859-1")
nextMW = playing_statistics_1920.iloc[-1]['MW'] + 1
nextMWFixtures = fixtures[fixtures.MW == nextMW]

playing_statistics_1920 = pd.concat([playing_statistics_1920,
                                     nextMWFixtures], ignore_index=True)


#define functions which return goals statistics

def get_goalsScored(playingStatistic):
    teams = {}
    for i in playingStatistic.groupby('HomeTeam').mean().T.columns:
        teams[i] = []
    
    for i in range(len(playingStatistic)):
        HTGS = playingStatistic.iloc[i]['FTHG']
        ATGS = playingStatistic.iloc[i]['FTAG']
        teams[playingStatistic.iloc[i].HomeTeam].append(HTGS)
        teams[playingStatistic.iloc[i].AwayTeam].append(ATGS)
    
    #create a dataframe where rows are teams and columns are matchweek
    
    GoalsScored = pd.DataFrame.from_dict(teams, orient='index')
    
    for i in range(1,len(GoalsScored.columns)):
        GoalsScored[i] = GoalsScored[i] + GoalsScored[i-1]
        
    return GoalsScored

def get_goalsConceded(playingStatistic):
    teams = {}
    for i in playingStatistic.groupby('HomeTeam').mean().T.columns:
        teams[i] = []
    
    for i in range(len(playingStatistic)):
        HTGC = playingStatistic.iloc[i]['FTAG']
        ATGC = playingStatistic.iloc[i]['FTHG']
        teams[playingStatistic.iloc[i].HomeTeam].append(HTGC)
        teams[playingStatistic.iloc[i].AwayTeam].append(ATGC)
    
    #create a dataframe where rows are teams and columns are matchweek
    
    GoalsConceded = pd.DataFrame.from_dict(teams, orient='index')
    
    for i in range(1,len(GoalsConceded.columns)):
        GoalsConceded[i] = GoalsConceded[i] + GoalsConceded[i-1]
    return GoalsConceded

def get_goals(playingStatistic):
    GC = get_goalsConceded(playingStatistic)
    GS = get_goalsScored(playingStatistic)
    
    j = 0
    HTGS = []
    ATGS = []
    HTGC = []
    ATGC = []
    
    for i in range(len(playingStatistic)):
        ht = playingStatistic.iloc[i].HomeTeam
        at = playingStatistic.iloc[i].AwayTeam
        
        if j == 0:
            HTGS.append(0)
            ATGS.append(0)
            HTGC.append(0)
            ATGC.append(0)
        else:
            HTGS.append(GS.loc[ht][j-1])
            ATGS.append(GS.loc[at][j-1])
            HTGC.append(GC.loc[ht][j-1])
            ATGC.append(GC.loc[at][j-1])
        
        if ((i+1) % 9) == 0:
            j = j+1
    
    playingStatistic['HTGS'] = HTGS
    playingStatistic['ATGS'] = ATGS
    playingStatistic['HTGC'] = HTGC
    playingStatistic['ATGC'] = ATGC
    
    return playingStatistic


print('Getting goals stats')
playing_statistics_1920 = get_goals(playing_statistics_1920)


#get current league points

def get_points(result):
    if result == 'W':
        return 3
    elif result == 'D':
        return 1
    else:
        return 0

def get_cum_points(matchres):
    matchres_points = matchres.applymap(get_points)
    for i in range(1,len(matchres_points.columns)):
        matchres_points[i] = matchres_points[i] + matchres_points[i-1]
    return matchres_points

def get_matchres(playing_stat):
    teams = {}
    for i in playing_stat.groupby('HomeTeam').mean().T.columns:
        teams[i] = []
    
    for i in range(len(playing_stat)):
        if playing_stat.iloc[i].FTR == 'H':
            teams[playing_stat.iloc[i].HomeTeam].append('W')
            teams[playing_stat.iloc[i].AwayTeam].append('L')
        elif playing_stat.iloc[i].FTR == 'A':
            teams[playing_stat.iloc[i].HomeTeam].append('L')
            teams[playing_stat.iloc[i].AwayTeam].append('W')
        else:
            teams[playing_stat.iloc[i].HomeTeam].append('D')
            teams[playing_stat.iloc[i].AwayTeam].append('D')
    
    return pd.DataFrame.from_dict(teams, orient='index')

def get_agg_points(playing_stat):
    matchres = get_matchres(playing_stat)
    cum_pts = get_cum_points(matchres)
    HTP = []
    ATP = []
    j = 0
    for i in range(len(playing_stat)):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        
        if j == 0:
            HTP.append(0)
            ATP.append(0)
        else:
            HTP.append(cum_pts.loc[ht][j-1])
            ATP.append(cum_pts.loc[at][j-1])
            
        if ((i + 1)% 9) == 0:
            j = j+1
    
    playing_stat['HTP'] = HTP
    playing_stat['ATP'] = ATP
    return playing_stat


print('Getting Points Stats')
playing_statistics_1920 = get_agg_points(playing_statistics_1920)


#get home advantage

def get_home_matchres(playing_statistic):
    hometeams = {}
    for i in playing_statistic.groupby('HomeTeam').mean().T.columns:
        hometeams[i] = []
    
    for i in range(len(playing_statistic)):
        if playing_statistic.iloc[i].FTR == 'H':
            hometeams[playing_statistic.iloc[i].HomeTeam].append(1)
        else:
            hometeams[playing_statistic.iloc[i].HomeTeam].append(0)
    
    df = pd.DataFrame.from_dict(hometeams, orient='index').T
    dfsum = df.cumsum()
    index = dfsum.index.to_series()
    mw = index + 1
    dfbyMW = dfsum.div(mw, axis=0)
    
    return dfbyMW

def get_away_matchres(playing_statistic):
    awayteams = {}
    for i in playing_statistic.groupby('AwayTeam').mean().T.columns:
        awayteams[i] = []
    
    for i in range(len(playing_statistic)):
        if playing_statistic.iloc[i].FTR == 'A':
            awayteams[playing_statistic.iloc[i].AwayTeam].append(1)
        else:
            awayteams[playing_statistic.iloc[i].AwayTeam].append(0)
    
    df = pd.DataFrame.from_dict(awayteams, orient='index').T
    dfsum = df.cumsum()
    index = dfsum.index.to_series()
    mw = index + 1
    dfbyMW = dfsum.div(mw, axis=0)
    return dfbyMW

def get_home_adv(playing_statistics):
    homewinPCT = get_home_matchres(playing_statistics).T
    awaywinPCT = get_away_matchres(playing_statistics).T
    
    homeTeamPCT = []
    awayTeamPCT = []
    j = 0
    
    for i in range(len(playing_statistics)):
        ht = playing_statistics.iloc[i].HomeTeam
        at = playing_statistics.iloc[i].AwayTeam
        
        if j == 0:
            homeTeamPCT.append(0)
            awayTeamPCT.append(0)
        else:        
            homeTeamPCT.append(homewinPCT.loc[ht][j-1])
            awayTeamPCT.append(awaywinPCT.loc[at][j-1])
        
        if((i+1)% 18) == 0:
            j = j + 1
    
    playing_statistics['HTHWPCT'] = homeTeamPCT
    playing_statistics['ATAWPCT'] = awayTeamPCT
    
    return playing_statistics

print('Getting home/away advantage')
playing_statistics_1920 = get_home_adv(playing_statistics_1920)

#get team form

def get_form(playing_stats):
    matchres = get_matchres(playing_stats)  
    matchres.fillna(value='M', inplace=True)
    
    j = 0
    htform3 = []
    atform3 = []
    htform5 = []
    atform5 = []
    
    for i in range(len(playing_stats)):
        ht = playing_stats.iloc[i].HomeTeam
        at = playing_stats.iloc[i].AwayTeam        
        
        if j == 0:
            htform3.append('M')
            atform3.append('M')
            htform5.append('M')
            atform5.append('M')            
        elif j == 1:
            matchres_last3 = matchres[[j-1]]
            last3sum = matchres_last3.cumsum(1)
            htform3.append(last3sum.loc[ht][j-1])
            atform3.append(last3sum.loc[at][j-1])
            htform5.append(last3sum.loc[ht][j-1])
            atform5.append(last3sum.loc[at][j-1])            
        elif j == 2:
            matchres_last3 = matchres[[j-2,j-1]]
            last3sum = matchres_last3.cumsum(1)
            htform3.append(last3sum.loc[ht][j-1])
            atform3.append(last3sum.loc[at][j-1])
            htform5.append(last3sum.loc[ht][j-1])
            atform5.append(last3sum.loc[at][j-1])            
        elif j == 3:
            matchres_last3 = matchres[[j-3,j-2,j-1]]
            last3sum = matchres_last3.cumsum(1)
            htform3.append(last3sum.loc[ht][j-1])
            atform3.append(last3sum.loc[at][j-1])            
            htform5.append(last3sum.loc[ht][j-1])
            atform5.append(last3sum.loc[at][j-1])
        elif j == 4:
            matchres_last3 = matchres[[j-3,j-2,j-1]]
            last3sum = matchres_last3.cumsum(1)
            htform3.append(last3sum.loc[ht][j-1])
            atform3.append(last3sum.loc[at][j-1])            
            matchres_last5 = matchres[[j-4,j-3,j-2,j-1]]
            last5sum = matchres_last5.cumsum(1)
            htform5.append(last5sum.loc[ht][j-1])
            atform5.append(last5sum.loc[at][j-1])
        else:
            matchres_last3 = matchres[[j-3,j-2,j-1]]
            last3sum = matchres_last3.cumsum(1)
            htform3.append(last3sum.loc[ht][j-1])
            atform3.append(last3sum.loc[at][j-1])            
            matchres_last5 = matchres[[j-5,j-4,j-3,j-2,j-1]]
            last5sum = matchres_last5.cumsum(1)
            htform5.append(last5sum.loc[ht][j-1])
            atform5.append(last5sum.loc[at][j-1])
        
        
        if((i+1)% 9) == 0:
            j = j + 1        
        
    playing_stats['HTF3'] = htform3
    playing_stats['ATF3'] = atform3
    playing_stats['HTF5'] = htform5
    playing_stats['ATF5'] = atform5
    
    
    return playing_stats

print('Getting Home/Away Team Form')
playing_statistics_1920 = get_form(playing_statistics_1920)

Standings = pd.read_csv('C:/Users/Michael/Desktop/FPLPredictionsWebsite/MachineLearningEPL/Bundesliga/BundesligaStandings.csv',encoding = "ISO-8859-1")
Standings.set_index(['Team'], inplace=True)
Standings = Standings.fillna(18)

def get_last(playing_stat, Standings, year):
    HomeTeamLP = []
    AwayTeamLP = []
    for i in range(len(playing_stat)):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HomeTeamLP.append(Standings.loc[ht][year])
        AwayTeamLP.append(Standings.loc[at][year])
    playing_stat['HomeTeamLP'] = HomeTeamLP
    playing_stat['AwayTeamLP'] = AwayTeamLP
    return playing_stat

print('Getting last seasons standings')
playing_statistics_1920 = get_last(playing_statistics_1920,Standings,19)

#final dataframe 

playing_stat = playing_statistics_1920

def get_form_points(string):
    sum = 0
    for letter in string:
        sum += get_points(letter)
    return sum

print('Getting form points')
playing_stat['HTF3Pts'] = playing_stat['HTF3'].apply(get_form_points)
playing_stat['ATF3Pts'] = playing_stat['ATF3'].apply(get_form_points)
playing_stat['HTF5Pts'] = playing_stat['HTF5'].apply(get_form_points)
playing_stat['ATF5Pts'] = playing_stat['ATF5'].apply(get_form_points)

print('Getting goal Difference, Points Difference, Difference in last seasons standings and difference in home/away record')
#get goal difference
playing_stat['HTGD'] = playing_stat['HTGS'] - playing_stat['HTGC']
playing_stat['ATGD'] = playing_stat['ATGS'] - playing_stat['ATGC']
#get difference in points
playing_stat['DiffPoints'] = playing_stat['HTP'] - playing_stat['ATP']
playing_stat['DiffFormPts3'] = playing_stat['HTF3Pts'] - playing_stat['ATF3Pts']
playing_stat['DiffFormPts5'] = playing_stat['HTF5Pts'] - playing_stat['ATF5Pts']
#difference in last years position
playing_stat['DiffLP'] = playing_stat['HomeTeamLP'] - playing_stat['AwayTeamLP']
#hometeam advantance
playing_stat['HomeAdv'] = playing_stat['HTHWPCT'] - playing_stat['ATAWPCT']


#Scale the bits that need scaling matter
cols = ['HTGS','ATGS','HTGC','ATGC','HTGD','ATGD','DiffPoints','HTP','ATP']
playing_stat.MW = playing_stat.MW.astype(float) 
for col in cols:
    playing_stat[col] = playing_stat[col] / playing_stat.MW


print('Finalising Datasets')
currentSeason = playing_stat[playing_stat.MW != nextMW]
nextWeek = playing_stat[playing_stat.MW == nextMW]

print('TIME TO MAKE PREDICTIONS')

completedSeasons = pd.read_csv('C:/Users/Michael/Desktop/FPLPredictionsWebsite/MachineLearningEPL/Bundesliga/CompletedSeasons.csv')

#this is the prediction engine for the data

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

predictions = predictResults(completedSeasons, currentSeason, nextWeek)

nextWeek['Prediction'] = predictions
nextWeek[['Date','HomeTeam','AwayTeam','Prediction']].to_json('C:/Users/Michael/Desktop/FPLPredictionsWebsite/MachineLearningEPL/Bundesliga/Prediction.json',orient='values')

print('Preditions Made')
