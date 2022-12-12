#!/usr/bin/env python
# coding: utf-8

# In[137]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectFromModel, SelectPercentile
from sklearn.metrics import f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB


# In[138]:


pd.set_option('display.max_columns', None)
df2 = pd.read_csv("hotel_bookings 4.csv")


# In[140]:


df2.isna().sum()


# In[141]:


df = pd.read_csv("hotel_bookings 4.csv")


# In[142]:


#Selecting numerical columns and checking summary statistics
num_cols = df.select_dtypes('number').columns

df[num_cols].describe().T


# In[151]:


df.describe()


# In[152]:


is_can = len(df[df['is_canceled']==1])
print("Percentage cancelation= ", is_can/len(df))
df['reservation_status'].value_counts(normalize=True)*100


# In[153]:


corr= df.corr(method='pearson')['is_canceled'][:]
corr


# In[154]:


sns.countplot(data=df, x='hotel', hue='is_canceled')
resort_canceled = df[(df['hotel']=='Resort Hotel') & (df['is_canceled']==1)]
city_canceled = df[(df['hotel']=='City Hotel') & (df['is_canceled']==1)]
print('Cancelations in resort hotel= ', (len(resort_canceled))/(len(df[df['hotel']=='Resort Hotel'])))
print('Cancelations in city hotel= ', (len(city_canceled))/(len(df[df['hotel']=='City Hotel'])))


# In[155]:


grid = sns.FacetGrid(df, col='is_canceled')
grid.map(plt.hist, 'lead_time', width=50)
grid.add_legend()


# In[156]:


print(len(df[(df['stays_in_weekend_nights']==0) & (df['stays_in_week_nights']==0)])) 


# In[157]:


((len(df.loc[(df['children']!=0) | (df['babies']!=0)]))/(len(df))) * 100


# In[158]:


sns.countplot(data=df, x='is_repeated_guest', hue='is_canceled')
new_guest = df[(df['is_repeated_guest']==0) & (df['is_canceled']==1)]
old_guest = df[(df['is_repeated_guest']==1) & (df['is_canceled']==1)]
print('Cancelations among new guests= ', (len(new_guest))/(len(df[df['is_repeated_guest']==0])))
print('Cancelations among old guests= ', (len(old_guest))/(len(df[df['is_repeated_guest']==1])))


# In[159]:


month_map = {'January':'01', 'February':'02', 'March':'03', 'April':'04', 'May':'05', 'June':'06', 'July':'07', 'August':'08', 'September':'09', 'October':'10', 'November':'11', 'December':'12'}
df.arrival_date_month = df.arrival_date_month.map(month_map).astype(int)


# In[160]:


df['arrival_date'] = df['arrival_date_year'].astype(str)+'-'+df['arrival_date_month'].astype(str)+'-'+df['arrival_date_day_of_month'].astype(str)


# In[161]:


def roomChange(row):
    if row['assigned_room_type'] == row['reserved_room_type']:
        return False
    else:
        return True

df['change_in_room'] = df.apply(roomChange, axis=1)


# In[162]:


df['children'] = df['children'].fillna(0)
df['offspring'] = (df['children'] + df['babies']).astype(int)


# In[163]:


df['total_bookings'] = df['previous_cancellations'] + df['previous_bookings_not_canceled']


# In[164]:


df['country'].fillna(df['country'].mode()[0], inplace=True)
df['agent'].fillna(df['agent'].mode()[0], inplace=True)
df['company'].fillna(df['company'].mode()[0], inplace=True)


# In[165]:


for i in range(len(df)):
    if df.loc[i, 'country'] == 'PRT':
        df.at[i, 'country'] = 1
    elif df.loc[i, 'country'] == 'GBR':
        df.at[i, 'country'] = 2
    else:
        df.at[i, 'country'] = 0


# In[166]:


df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
df['arrival_date'] = pd.to_datetime(df['arrival_date'])


# In[167]:


df['stay_duration'] = df['reservation_status_date'] - df['arrival_date']
df['stay_duration'] = df['stay_duration'] / np.timedelta64(1, 'D')
df['stay_duration'] = df['stay_duration'].astype(int)


# In[169]:


for i in range(len(df)):
    if df.loc[i, 'stay_duration']<0:
        df.at[i, 'stay_duration'] = -1


# In[170]:


lb = LabelEncoder()
var = ['hotel', 'customer_type', 'deposit_type', 'change_in_room', 'market_segment', 'distribution_channel', 'country']
for item in var:
    df[item] = lb.fit_transform(df[item])
df = pd.get_dummies(df, columns=['hotel', 'customer_type', 'deposit_type', 'change_in_room', 'market_segment', 'distribution_channel', 'country'])


# In[171]:


df.drop(['meal', 'assigned_room_type', 'reserved_room_type', 'reservation_status', 'reservation_status_date', 'arrival_date'], axis=1, inplace=True)


# In[172]:


train_var = df.drop(['is_canceled'], axis=1)
test_var = df['is_canceled']
X_train, X_test, y_train, y_test = train_test_split(train_var, test_var, test_size=0.20)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[202]:


y_train


# In[203]:


from collections import Counter

Counter(y_train).keys() # equals to list(set(words))
Counter(y_train).values() # counts the elements' frequency


# In[174]:


#Data scaling
std_scaler = StandardScaler()
std_scaler.fit(X_train)
X_train_std = std_scaler.transform(X_train)
X_test_std = std_scaler.transform(X_test)

mm_scaler = MinMaxScaler()
mm_scaler.fit(X_train)
X_train_mm = mm_scaler.transform(X_train)
X_test_mm = mm_scaler.transform(X_test)


# In[191]:


#Logistic Regression
logreg = LogisticRegression(max_iter=500).fit(X_train_mm, y_train)
scores = cross_val_score(logreg, X_train_mm, y_train, cv=5)
logreg_pred = logreg.predict(X_test_mm)
print("Average cross validation score: {:.3f}".format(scores.mean()))
print("Test accuracy: {:.3f}".format(logreg.score(X_test_mm, y_test)))
print("F1 score: {:.3f}".format(f1_score(y_test, logreg_pred)))
print(confusion_matrix(y_test, logreg_pred))


# In[176]:


model = LogisticRegression(max_iter=500)
# fit the model
model.fit(X_train_mm, y_train)
# get importance
importance = model.coef_[0]

df_lr = pd.DataFrame(
    {'feature': X_train.columns,
     'score': importance
    })

df_lr = df_lr.sort_values('score', ascending=False)
# df_lr['score'] = (df_lr['score'] - df_lr['score'].min()) / (df_lr['score'].max() - df_lr['score'].min()) 

ax = df_lr[0:10].plot.bar(x='feature', y='score', rot=90, color='maroon', title='Feature importance score of Logistic Regression model')
ax.set_xlabel("Features ")
ax.set_ylabel("Importance Score")


# In[177]:


#Linear SVC
svc = LinearSVC().fit(X_train_mm, y_train)
scores = cross_val_score(svc, X_train_mm, y_train, cv=5)
svc_pred = svc.predict(X_test_mm)
print("Average cross validation score: {:.3f}".format(scores.mean()))
print("Test accuracy: {:.3f}".format(svc.score(X_test_mm, y_test)))
print("F1 score: {:.3f}".format(f1_score(y_test, svc_pred)))
print(confusion_matrix(y_test, svc_pred))


# In[178]:


#SGD Classifier
sgd = SGDClassifier(alpha=0.1).fit(X_train_std, y_train)
scores = cross_val_score(sgd, X_train_std, y_train, cv=5)
sgd_pred = sgd.predict(X_test_std)
print("Average cross validation score: {:.3f}".format(scores.mean()))
print("Test accuracy: {:.3f}".format(sgd.score(X_test_std, y_test)))
print("F1 score: {:.3f}".format(f1_score(y_test, sgd_pred)))
print(confusion_matrix(y_test, sgd_pred))


# In[179]:


#Decision Tree
tree = DecisionTreeClassifier(max_depth=1).fit(X_train, y_train)
scores = cross_val_score(tree, X_train, y_train, cv=5)
tree_pred = tree.predict(X_test)
print("Average cross validation score: {:.3f}".format(scores.mean()))
print("Test accuracy: {:.3f}".format(tree.score(X_test, y_test)))
print("F1 score: {:.3f}".format(f1_score(y_test, tree_pred)))
print(confusion_matrix(y_test, tree_pred))


# In[180]:


clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
feat_importance = clf.tree_.compute_feature_importances(normalize=False)
print("feat importance = " + str(feat_importance))


# In[84]:


df_dt = pd.DataFrame(
    {'feature': X_train.columns,
     'score': feat_importance
    })

df_dt = df_dt.sort_values('score', ascending=False)


# In[103]:


df_dt['score'] = (df_dt['score'] - df_dt['score'].min()) / (df_dt['score'].max() - df_dt['score'].min())
df_dt[0:10].plot.bar(x='feature', y='score', rot=90, color='maroon', title='Feature importance score of Decision Tree model')
ax.set_xlabel("Features ")
ax.set_ylabel("Importance Score")


# In[30]:


def classifier(train, test, estimator, param_grid):
    grid_search = GridSearchCV(estimator, param_grid, cv=5)
    grid_search.fit(train, y_train)
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    print("Test score: {:.3f}".format(grid_search.score(test, y_test)))

def feature_selection(model):
    select_features = SelectFromModel(estimator=model, threshold='median')
    select_features.fit(X_train, y_train)
    X_train_select = select_features.transform(X_train)
    X_test_select = select_features.transform(X_test)
    return X_train_select, X_test_select

def run_model(model, model_feature, param_grid):
    print("Before feature selection:")
    classifier(X_train, X_test, model, param_grid)
    X_train_select, X_test_select = feature_selection(model_feature)
    print("After feature selection")
    classifier(X_train_select, X_test_select, model, param_grid)


# In[31]:


#Random Forest
param_grid = {'n_estimators':[50,75,100], 'max_depth':[1,2,5]}
run_model(RandomForestClassifier(), RandomForestClassifier(n_estimators=50, max_depth=2), param_grid)


# In[181]:


#Adaboost Classifier
ada = AdaBoostClassifier()
ada.fit(X_train, y_train)
ada_pred = ada.predict(X_test)
scores = cross_val_score(ada, X_train, y_train, cv=5)
print("Average cross validation score: {:.3f}".format(scores.mean()))
print("Test accuracy: {:.3f}".format(ada.score(X_test, y_test)))
print("F1 score: {:.3f}".format(f1_score(y_test, ada_pred)))
print(confusion_matrix(y_test, ada_pred))


# In[182]:


from sklearn.metrics import ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, ada_pred), display_labels=ada.classes_)
disp.plot()


# In[104]:


df_ada = pd.DataFrame(
    {'feature': X_train.columns,
     'score': ada.feature_importances_
    })

df_ada = df_ada.sort_values('score', ascending=False)

df_ada['score'] = (df_ada['score'] - df_ada['score'].min()) / (df_ada['score'].max() - df_ada['score'].min())
df_ada[0:10].plot.bar(x='feature', y='score', rot=90, color='maroon', title='Feature importance score of Adaboost model')
ax.set_xlabel("Features ")
ax.set_ylabel("Importance Score")


# In[119]:


model = RandomForestClassifier(max_depth= 5, n_estimators=100)
# fit the model
model.fit(X_train, y_train)
# get importance
importance = model.feature_importances_


# In[121]:


print("Average cross validation score: {:.3f}".format(scores.mean()))
print("Test accuracy: {:.3f}".format(model.score(X_test, y_test)))


# In[120]:


df_rf = pd.DataFrame(
    {'feature': X_train.columns,
     'score': importance
    })

df_rf = df_rf.sort_values('score', ascending=False)

df_rf['score'] = (df_rf['score'] - df_rf['score'].min()) / (df_rf['score'].max() - df_rf['score'].min())

df_rf[0:10].plot.bar(x='feature', y='score', rot=90, color='maroon', title='Feature importance score of Random Forest model')
ax.set_xlabel("Features ")
ax.set_ylabel("Importance Score")


# In[ ]:


print("Average cross validation score: {:.3f}".format(scores.mean()))
print("Test accuracy: {:.3f}".format(model.score(X_test, y_test)))


# In[195]:


from xgboost import XGBClassifier
from matplotlib import pyplot
import shap

# fit model no training data
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
# feature importance

df_xgb = pd.DataFrame(
    {'feature': X_train.columns,
     'score': xgb.feature_importances_
    })

df_xgb = df_xgb.sort_values('score', ascending=False)

df_xgb[0:10].plot.bar(x='feature', y='score', rot=90)


# In[132]:


mod = ['ADA Boost',
       'Random Forest',
'Decision Tree',
'Linear SVC',
'Logistic Regression',
'SGD Classifier']

f1 = [0.999, 0.994, 0.994, 0.994,0.987,0.983]

tst = [1,0.999, 0.993,0.995,0.991,0.988]

dfev = df_xgb = pd.DataFrame(
    {'Model': mod,
     'F1 score': f1,
     'Test Accuracy': tst
    })


# In[133]:


dfev


# In[186]:


from sklearn.metrics import accuracy_score
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

acc_knn = accuracy_score(y_test, y_pred_knn)
conf = confusion_matrix(y_test, y_pred_knn)
# clf_report = classification_report(y_test, y_pred_knn)

print(f"Accuracy Score of KNN is : {acc_knn}")
print(f"Confusion Matrix : \n{conf}")
print("F1 score: {:.3f}".format(f1_score(y_test, y_pred_knn)))


# In[187]:


dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred_dtc = dtc.predict(X_test)

acc_dtc = accuracy_score(y_test, y_pred_dtc)
conf = confusion_matrix(y_test, y_pred_dtc)
# clf_report = classification_report(y_test, y_pred_dtc)

print(f"Accuracy Score of Decision Tree is : {acc_dtc}")
print(f"Confusion Matrix : \n{conf}")
print("F1 score: {:.3f}".format(f1_score(y_test, y_pred_dtc)))

