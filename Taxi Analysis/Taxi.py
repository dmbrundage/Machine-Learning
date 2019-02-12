# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 19:31:53 2018

@author: dmbru
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import datetime
from datetime import date
import calendar
import xgboost as xgb
chunksize = 10**6

for chunk in pd.read_csv('train.csv', chunksize=chunksize):
    process(chunk)
test = pd.read_csv('test.csv')

chunk['dt'] = pd.to_datetime(chunk['pickup_datetime'])
test['dt'] = pd.to_datetime(test['pickup_datetime'])

chunk = chunk.drop(chunk[chunk.isnull().any(1)].index, axis = 0)
chunk = chunk.drop(chunk[chunk['fare_amount']<0].index, axis=0)

chunk['caldate'] = chunk.apply(lambda row:calendar.day_name[row['dt'].weekday()], axis=1)
test['caldate'] = test.apply(lambda row:calendar.day_name[row['dt'].weekday()], axis=1)

train = chunk

def haversine_distance(lat1, long1, lat2, long2):
    data = [train, test]
    for i in data:
        R = 6371  #radius of earth in kilometers
        #R = 3959 #radius of earth in miles
        phi1 = np.radians(i[lat1])
        phi2 = np.radians(i[lat2])
    
        delta_phi = np.radians(i[lat2]-i[lat1])
        delta_lambda = np.radians(i[long2]-i[long1])
    
        #a = sin²((φB - φA)/2) + cos φA . cos φB . sin²((λB - λA)/2)
        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    
        #c = 2 * atan2( √a, √(1−a) )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
        #d = R*c
        d = (R * c) #in kilometers
        i['H_Distance'] = d
    return d

haversine_distance('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')

y_train = train['fare_amount']
X_train = train.drop('fare_amount', axis=1)


def airport_pickup(row):
    if row['pickup_latitude'] == 40.6413 and row['pickup_longitude'] == 73.7781:
        return 1
    if row['pickup_latitude'] == 40.7769 and row['pickup_longitude'] == 73.8740:
        return 1
    return 0

X_train['airport_pickup']= X_train.apply(lambda row: airport_pickup(row),axis=1)

X_train = X_train.drop('caldate', axis=1)
test = test.drop('caldate', axis=1)

X_train = X_train.drop('dt', axis=1)
test = test.drop('dt', axis=1)

X_train = X_train.drop('pickup_datetime', axis=1)
test = test.drop('pickup_datetime', axis=1)

X_train = X_train.drop('key', axis=1)
test = test.drop('key', axis=1)

X_train[(X_train['pickup_latitude'] == 40.6413) & (X_train['pickup_longitude'] == 73.7781)]

data = [X_train,test]
for i in data:
    i['Year'] = i['dt'].dt.year
    i['Month'] = i['dt'].dt.month
    i['Date'] = i['dt'].dt.day
    i['Day of Week'] = i['dt'].dt.dayofweek
    i['Hour'] = i['dt'].dt.hour
    
    
rf = RandomForestRegressor()
params_rf = {
    'n_estimators':[100,350,500],
    'max_features':['log2','auto','sqrt'],
    'min_samples_leaf':[2,10,30]
}

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Instantiate grid_rf
grid_rf = GridSearchCV(estimator=rf,
                       param_grid=params_rf,
                       cv=3,
                       n_jobs=1)

grid_rf.fit(X_train, y_train)

best_model = grid_rf.best_estimator__

rf.fit(X_train, y_train)

pred = rf.predict(test)

testkey = test['key']

submission = pd.Series.to_frame(testkey)
submission['fare_amount'] = pred.tolist()

submission = pd.read_csv('sample_submission.csv')
submission['fare_amount'] = pred
submission.to_csv('submission_1.csv', index=False)
