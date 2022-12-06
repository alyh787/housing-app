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

#import dataset
housing_df = pd.read_csv("austinHousingData.csv")
df = housing_df.drop(columns=['description','numOfAccessibilityFeatures','homeImage'])
