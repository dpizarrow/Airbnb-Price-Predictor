import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import pickle

import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('dataset/AB_NYC_2019.csv')

df['reviews_per_month'].fillna(df['reviews_per_month'].mean(), inplace=True)


for i in range(0,len(df)):
    if df['room_type'][i]=='Private room':
        df['room_type'][i]=1
    elif df['room_type'][i]=='Entire home/apt':
        df['room_type'][i]=2
    else:
        df['room_type'][i]=3     



for i in range(0,len(df)):
    if df['neighbourhood_group'][i]=='Brooklyn':
        df['neighbourhood_group'][i]=1
    else:
        df['neighbourhood_group'][i]=2    



# Drop unnecessary columns

df.drop(['id','name','host_id','host_name','last_review','latitude','longitude','reviews_per_month','neighbourhood', 'number_of_reviews', 'calculated_host_listings_count'], axis=1, inplace=True)



df = df.dropna()


df.apply(pd.to_numeric)



# Split data into train and test sets


Y = df['price']
X = df.drop('price', axis=1)

print(X.columns)

X = X.apply(pd.to_numeric, errors='coerce')
Y = Y.apply(pd.to_numeric, errors='coerce')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Train model

model = LinearRegression()
model.fit(X_train, Y_train)

predictions = model.predict(X_test)

print(f'LinearRegression Mean Squared Error: {mean_squared_error(predictions, Y_test)}')
print(f'LinearRegression R2 Score: {r2_score(Y_test, predictions)}')


# Ridge Model

ridge = Ridge(normalize=True)
params = {'alpha': [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor = GridSearchCV(ridge, params, scoring='neg_mean_squared_error', cv=5)
ridge_regressor.fit(X_train, Y_train)
ridge_preds = ridge_regressor.predict(X_test)

print(f'Ridge Mean Squared Error: {mean_squared_error(ridge_preds, Y_test)}')
print(f'Ridge R2 Score: {r2_score(Y_test, ridge_preds)}')

# Lasso

lasso = Lasso(normalize=True)
params = {'alpha': [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor = GridSearchCV(lasso, params, scoring='neg_mean_squared_error', cv=5)
lasso_regressor.fit(X_train, Y_train)
lasso_preds = lasso_regressor.predict(X_test)

print(f'Lasso Mean Squared Error: {mean_squared_error(lasso_preds, Y_test)}')
print(f'Lasso R2 Score: {r2_score(Y_test, lasso_preds)}')

# Save model

pickle.dump(lasso_regressor, open('model.pkl', 'wb'))

# Load model

model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[2,9,6,6]]))