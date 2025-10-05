from sklearn import linear_model
import pandas as pd
import pickle


df= pd.read_csv(r'C:\Users\annsh\Downloads\SPGISummer2018-FlaskTutorial-master\SPGISummer2018-FlaskTutorial-master\model\price.csv')

y = df['Value'] 
X = df[['Rooms', 'Distance']]

lm = linear_model.LinearRegression()
lm.fit(X, y)

pickle.dump(lm, open('model.pkl','wb')) 

print(lm.predict([[15, 61]]))
print(f'score:{lm.score(X,y)}')