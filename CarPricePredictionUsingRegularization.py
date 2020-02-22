import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('G:/Software/Machine learning/1/24. Advanced Machine Learning Algorithms/AdvanceReg/CarPrice_Assignment.csv')

print(df.isnull().sum())
print(df.info())


print(df['symboling'].astype('category').value_counts())
print(df['drivewheel'].astype('category').value_counts())
print(df['CarName'].astype('category').value_counts())
print(df['enginelocation'].astype('category').value_counts())
print(df['horsepower'].astype('category').value_counts())


sns.distplot(df['symboling'])
plt.show()

sns.distplot(df['wheelbase'])
plt.show()


sns.distplot(df['enginesize'])
plt.show()
 
sns.distplot(df['boreratio'])
plt.show()


sns.distplot(df['stroke'])
plt.show()

sns.distplot(df['compressionratio'])
plt.show()


sns.distplot(df['citympg'])
plt.show()

sns.distplot(df['highwaympg'])
plt.show()


sns.distplot(df['price'])
plt.show()



cars_numeric_value = df.select_dtypes(include = ['float64' , 'int64'])
cars_numeric_value.head()


cars_numeric_value = cars_numeric_value.drop(['symboling' , 'car_ID'] , axis = 1)
cars_numeric_value.head()


plt.figure(figsize = (20,10))
sns.pairplot(cars_numeric_value)
plt.show()


cor = cars_numeric_value.corr()
print(cor)


plt.figure(figsize=(16,8))
sns.heatmap(cor, cmap="YlGnBu", annot=True)
plt.show()

print(df.info())
df['symboling'] = df['symboling'].astype('object')
df.info()


carnames = df['CarName'].apply(lambda x: x.split(" ")[0])
print(carnames)



import re
p = re.compile(r'\w+-?\w+')
carnames = df['CarName'].apply(lambda x: re.findall(p, x)[0])
print(carnames)


df['CarCompany'] = df['CarName'].apply(lambda x: re.findall(p , x)[0])
print(df['CarCompany'].astype('category').value_counts())



df.loc[(df['CarCompany'] == 'vw') | (df['CarCompany'] == 'vokswagen') , 'CarCompany'] = 'volkswagen'
df.loc[(df['CarCompany'] == 'toyouta') , 'CarCompany'] = 'toyota'
df.loc[(df['CarCompany'] == 'porcshce') , 'CarCompany'] = 'porsche'
df.loc[df['CarCompany'] == "Nissan", 'car_company'] = 'nissan'
df.loc[df['CarCompany'] == "maxda", 'car_company'] = 'mazda'

print(df['CarCompany'].astype('category').value_counts())


df = df.drop('CarName', axis=1)
print(df.info())


x = df.drop(['price' , 'car_ID'] , axis = 1)
y = df['price']


x_categorical = x.select_dtypes(include=['object'])
car_dummies = pd.get_dummies(x_categorical , drop_first = True)
x = x.drop(list(x_categorical.columns) , axis = 1)
x = pd.concat([x , car_dummies] , axis = 1)



from sklearn.preprocessing import scale
cols = x.columns
x = pd.DataFrame(scale(x))
x.columns = cols
print(x.columns)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x , y , test_size = 0.3, random_state = 100)




'''Ridge'''
params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}


ridge = Ridge()
folds = 5
gd = GridSearchCV(estimator = ridge , param_grid = params , scoring = 'neg_mean_absolute_error' , cv = folds , return_train_score = True)
gd.fit(X_train , y_train)


cv_results = pd.DataFrame(gd.cv_results_)
cv_results = cv_results[cv_results['param_alpha']<=200]


cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')
plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()



alpha = 15
ridge = Ridge(alpha=alpha)

ridge.fit(X_train, y_train)
ridge.coef_



'''Lasso'''
lasso = Lasso()
model_cv = GridSearchCV(estimator = lasso,  param_grid = params, scoring= 'neg_mean_absolute_error',  cv = folds,  return_train_score=True,verbose = 1)            
model_cv.fit(X_train, y_train) 


cv_results_lasso = pd.DataFrame(model_cv.cv_results_)

cv_results_lasso['param_alpha'] = cv_results_lasso['param_alpha'].astype('float32')


plt.plot(cv_results_lasso['param_alpha'], cv_results_lasso['mean_train_score'])
plt.plot(cv_results_lasso['param_alpha'], cv_results_lasso['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')
plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()


alpha =100
lasso = Lasso(alpha=alpha)
lasso.fit(X_train, y_train) 
print(lasso.coef_)

y_ls = lasso.predict(X_test)
ac = r2_score(y_test, y_ls)
print(ac*100)