#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import folium
from folium import plugins
import seaborn as sns

import streamlit as st

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score

#load dataset
housing_df = pd.read_csv("austinHousingData.csv")
df = housing_df.drop(columns=['description','numOfAccessibilityFeatures','homeImage'])

st.write("""
# Austin Housing Price Prediction Tool
""")
st.write('Please select input parameters in the sidebar')
st.write('---')

## start ML model -- pick out primary variables

primary_variables = ['numOfHighSchools',
 'longitude',
 'numOfPrimarySchools',
 'zipcode',
 'numOfWaterfrontFeatures',
 'numOfMiddleSchools',
 'hasView',
 'numOfPhotos',
 'numOfElementarySchools',
 'parkingSpaces',
 'garageSpaces',
 'hasSpa',
 'MedianStudentsPerTeacher',
 'numOfStories',
 'avgSchoolRating',
 'numOfBedrooms',
 'livingAreaSqFt',
 'numOfBathrooms']
#split df into input and output variables
X = df[primary_variables].astype(float)
y = df['latestPrice']
#split into training and testing set
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y)
#Load random forest model and train
clf = RandomForestRegressor(n_estimators=50,max_features=6)
clf.fit(Xtrain,ytrain)

st.sidebar.header('Input Parameters')

def userinput():
    numOfHighSchools = st.sidebar.slider('Number of High schools',X.numOfHighSchools.min(),X.numOfHighSchools.max(),float(X.numOfHighSchools.mean()),step=float(1))
    longitude = st.sidebar.slider('Longitude',X.longitude.min(),X.longitude.max(),float(X.longitude.mean()))
    numOfPrimarySchools = st.sidebar.slider('Number of Primary schools',X.numOfPrimarySchools.min(),X.numOfPrimarySchools.max(),float(X.numOfPrimarySchools.mean()),step=float(1))
    zipcode = st.sidebar.slider('zipcode',X.zipcode.min(),X.zipcode.max(),float(X.zipcode.mean()),step=float(1))
    numOfWaterfrontFeatures = st.sidebar.slider('Number of Waterfront Features',X.numOfWaterfrontFeatures.min(),X.numOfWaterfrontFeatures.max(),float(X.numOfWaterfrontFeatures.mean()),step=float(1))
    numOfMiddleSchools = st.sidebar.slider('Number of Middle schools',X.numOfMiddleSchools.min(),X.numOfMiddleSchools.max(),float(X.numOfMiddleSchools.mean()),step=float(1))
    hasView = st.sidebar.slider('Has View?',X.hasView.min(),X.hasView.max(),float(X.hasView.mean()),step=float(.5))
    numOfPhotos = st.sidebar.slider('Number of Photos in Listing',X.numOfPhotos.min(),X.numOfPhotos.max(),float(X.numOfPhotos.mean()),step=float(1))
    numOfElementarySchools = st.sidebar.slider('Number of Elementary schools',X.numOfElementarySchools.min(),X.numOfElementarySchools.max(),float(X.numOfElementarySchools.mean()),step=float(1))
    parkingSpaces = st.sidebar.slider('Parking Spaces',X.parkingSpaces.min(),X.parkingSpaces.max(),float(X.parkingSpaces.mean()),step=float(.5))
    garageSpaces = st.sidebar.slider('Garage Spaces',X.garageSpaces.min(),X.garageSpaces.max(),float(X.garageSpaces.mean()),step=float(.5))
    hasSpa = st.sidebar.slider('Has Spa?',X.hasSpa.min(),X.hasSpa.max(),float(X.hasSpa.mean()),step=float(.5))
    MedianStudentsPerTeacher = st.sidebar.slider('Median Students per Teacher',X.MedianStudentsPerTeacher.min(),X.MedianStudentsPerTeacher.max(),float(X.MedianStudentsPerTeacher.mean()))
    numOfStories = st.sidebar.slider('Number of Stories',X.numOfStories.min(),X.numOfStories.max(),float(X.numOfStories.mean()),step=float(1))
    avgSchoolRating = st.sidebar.slider('Average School Rating',X.avgSchoolRating.min(),X.avgSchoolRating.max(),float(X.avgSchoolRating.mean()))
    numOfBedrooms = st.sidebar.slider('Number of Bedrooms',X.numOfBedrooms.min(),X.numOfBedrooms.max(),float(X.numOfBedrooms.mean()),step=float(1))
    livingAreaSqFt = st.sidebar.slider('Living room Area [sq.ft]',X.livingAreaSqFt.min(),X.livingAreaSqFt.max(),float(X.livingAreaSqFt.mean()))
    numOfBathrooms = st.sidebar.slider('Number of Bathrooms',X.numOfBathrooms.min(),X.numOfBathrooms.max(),float(X.numOfBathrooms.mean()),step=float(1))
    data = {'numOfHighSchools':numOfHighSchools,
             'longitude':longitude,
             'numOfPrimarySchools':numOfPrimarySchools,
             'zipcode':zipcode,
             'numOfWaterfrontFeatures':numOfWaterfrontFeatures,
             'numOfMiddleSchools':numOfMiddleSchools,
             'hasView':hasView,
             'numOfPhotos':numOfPhotos,
             'numOfElementarySchools':numOfElementarySchools,
             'parkingSpaces':parkingSpaces,
             'garageSpaces':garageSpaces,
             'hasSpa':hasSpa,
             'MedianStudentsPerTeacher':MedianStudentsPerTeacher,
             'numOfStories':numOfStories,
             'avgSchoolRating':avgSchoolRating,
             'numOfBedrooms':numOfBedrooms,
             'livingAreaSqFt':livingAreaSqFt,
             'numOfBathrooms':numOfBathrooms}
    features = pd.DataFrame(data,index=[0])
    return features

df1 = userinput()

st.header('Specified Input Parameters')
st.write(df1)
st.write('---')

##predict value based on user input
prediction = clf.predict(df1)

st.header('Predicted House Value')
st.write(prediction)
st.write('---')