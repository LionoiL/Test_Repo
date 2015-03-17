import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

# import os
# print(os.getcwd())
# os.chdir("..")
# print(os.getcwd())

def featureEngineer(df):

    print(list(df))
    print(df.dtypes)


    # NOTE ON USING CATEGORICAL VARIABLES: It turns out that RFs in sklearn CANNOT deal with categorical
    # variables (this is despite some of the sklearn documentation claiming that it does). The solution
    # is to use one hot hashing, which is done with sklearn.preprocessing.OneHotEncoder

    # NOTE WHICH VARIABLES TO MAKE CATEGORICAL:
    # in the R-code season, holiday, workingday and weather were all stored as categorical variables.
    # In this current version 'm leaving them as numericals. We can try later whether using one-hot
    # encoding improves things

    df['hour'] = map(lambda x: x[11:13], df.iloc[:, 0]) #create hour column
    df['month'] = map(lambda x: x[5:7], df.iloc[:, 0]) #create month column
    df['year'] = map(lambda x: x[0:4], df.iloc[:, 0]) #create year column
    df['weekday'] = map(lambda x: time.strftime("%A", time.strptime(x[0:10], "%Y-%m-%d")), df.iloc[:,0])

    enc = OneHotEncoder()

    return(df)


## LOADS TRAINING AND TEST DATA AND ADD TIME FEATURES
train = pd.read_csv("train.csv")
train = featureEngineer(train)

test = pd.read_csv("test.csv")
featureEngineer(test)

testTrain = train #testTrain: The training variables to train the RF on
#remove the bike rental columns
testTrain.drop(testTrain[["datetime", "count", "registered"]] , axis=1, inplace=True)
print(testTrain)

rf = RandomForestClassifier(n_estimators=2000,  max_depth=None, min_samples_split=1, random_state=0)
casualFit = rf.fit( testTrain, train[['casual']])

# print(casualFit)




#TO DO: Might want to redo the importance analysis presently done in R, because presumably we'll be
#       adding more features

# print(train[['datetime']])
# print(train[['hour']])
# print(train[['month']])
# print(train[['year']])
# print(train[['weekday']])

# print(train.iloc[0,0].weekday())
# print (time.strftime("%A", time.strptime(train.iloc[0,0][0:10], "%Y-%m-%d")))
