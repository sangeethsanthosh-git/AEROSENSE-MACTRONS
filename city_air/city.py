



#importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import pickle


import warnings
warnings.filterwarnings('ignore')

pest=pd.read_csv(r'C:\other\Airpolution_2023\SPGISummer2018-FlaskTutorial-master\city_air\city_day.csv')



df = pest.copy()

df.info()

def preprocessing(df):
    df['Date'] = pd.to_datetime(df['Date'])
    
    df['Year'] = [d.year for d in df['Date']]
    df['Month'] = [d.month for d in df['Date']]
    
    df.dropna(subset = ['AQI'],inplace = True)
    df = df.drop(['Xylene'],axis = 1)
    
    return df

df = preprocessing(df)
df
#print(df)



cities = df['City'].unique()
cities
#print(cities)


jaipur = df[df['City'] == 'Amaravati']
delhi = df[df['City'] == 'Delhi']
guwahati = df[df['City'] == 'Guwahati']

def missing_values_table(df):
    table = pd.DataFrame(df.isnull().sum(), columns = ['count'])
    table['percentage'] = round((table['count']/df.shape[0])*100,2)
    return table
missing_values_table(delhi)
print(jaipur)
print(missing_values_table(jaipur))






