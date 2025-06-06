import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  This is complete EDA(Exploratory Data Analysis) on the dataset

df = pd.read_csv("notebooks/AlgerianForest_cleaned_dataset.csv")


df=df.drop(['day', 'month', 'year'], axis=1)


df['Classes']= np.where(df['Classes'].str.contains('not fire'),0,1)



# DISTRIBUTION OF FIRE VS NONFIRE DATA

sns.set_theme(style="darkgrid")  
df.hist(bins=50, figsize=(20, 15))
plt.show()
print(df['Classes'].value_counts(normalize=True)*100)
labels=['Not Fire', 'Fire']
sizes=df['Classes'].value_counts(normalize=True)*100
colors=['orange', 'blue']
plt.figure(figsize=(8,8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, explode=(0.05, 0.05), shadow=True)
plt.title('Fire vs Not Fire Distribution')
plt.axis('equal')
plt.show()
print(df.corr())
sns.heatmap(df.corr())
plt.show()
sns.boxplot(df['FWI'], color='green')



# VISUALIZATION OF MONTHLY FIRE DISTRIBUTION


df_org=pd.read_csv("notebooks/Algerian_forest_fires_dataset.csv")
df_org.columns=df_org.columns.str.strip()
df_org['Classes']= np.where(df_org['Classes'].str.contains('not fire'),'not fire','fire')
print(df_org.head())
dftemp=df.loc[df['Region']==1]
sns.set_style("whitegrid")
sns.countplot(x='month', hue='Classes', data= df_org, palette={'fire':'red', 'not fire':'green'})
plt.ylabel('Number of Fires')
plt.xlabel('Month')
plt.title('Fire Analysis of Sidi-bel Region')
plt.show()



x = df.drop(['FWI'], axis=1)
y = df['FWI']


#  The below function is used to find the multicollinearity in the input features and remove it.
def correlation(dataset, thresh):
    corrcol=set()
    corrmat=dataset.corr()
    for i in range(len(corrmat.columns)):
        for j in range (i):
            if abs(corrmat.iloc[i,j]>thresh):
                corrcol.add(corrmat.columns[i])
    return corrcol
# print(correlation(x, 0.85))

# Removing the multicollinearity features
x.drop(correlation(x, 0.85), axis=1, inplace=True)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.25, random_state=42)

from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
x_train_scaled=scalar.fit_transform(x_train)
x_test_scaled=scalar.transform(x_test)


# Visualizing the effect of scaling on the data

plt.subplot(1,2,1)
sns.boxplot(data=x_train)
plt.title('Before Scaling')
plt.subplot(1,2,2)
sns.boxplot(data=x_train_scaled)
plt.title('After Scaling')
plt.show()



# This is Linear Regression
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train_scaled, y_train)
y_pred=regression.predict(x_test_scaled)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print("Linear Regression Results:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))    
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R-squared in percentage:", 100*r2_score(y_test, y_pred),"%")



# This is Lasso Regression
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(x_train_scaled, y_train)
y_pred=lasso.predict(x_test_scaled)
print("\n\nLasso Regression Results:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))    
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R-squared in percentage:", 100*r2_score(y_test, y_pred),"%")



# This is Lasso with Cross Validation
from sklearn.linear_model import LassoCV
lassocv=LassoCV(cv=5)
lassocv.fit(x_train_scaled, y_train)
y_pred=lassocv.predict(x_test_scaled)
print("\n\nLasso with Cross Validation Results:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R-squared in percentage:", 100*r2_score(y_test, y_pred),"%")


# This is Ridge Regression
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(x_train_scaled, y_train)
y_pred=ridge.predict(x_test_scaled)
print("\n\nRidge with Cross Validation Results:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R-squared in percentage:", 100*r2_score(y_test, y_pred),"%")



# This is Ridge with Cross Validation
from sklearn.linear_model import RidgeCV
ridgecv=RidgeCV(cv=5)
ridgecv.fit(x_train_scaled, y_train)
y_pred=ridgecv.predict(x_test_scaled)
print("\n\nRidge with Cross Validation Results:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R-squared in percentage:", 100*r2_score(y_test, y_pred),"%")



# This is ElasticNet Regression
from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet()
elasticnet.fit(x_train_scaled, y_train)
y_pred=elasticnet.predict(x_test_scaled)
print("\n\nElasticNet Regression Results:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R-squared in percentage:", 100*r2_score(y_test, y_pred),"%")


# import pickle
# pickle.dump(scalar, open('scalar.pkl', 'wb'))
# pickle.dump(ridge, open('ridge.pkl', 'wb'))
