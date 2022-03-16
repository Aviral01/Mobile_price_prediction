# -*- coding: utf-8 -*-
"""## Load Data"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

dataset=pd.read_csv('D:/train.csv')

dataset.head()

"""## Data Analysis"""

dataset.info()

dataset.describe()

"""## Data Visualization & Analysis

### How does ram is affected by price
"""

sns.lineplot(x='ram',y='price_range',data=dataset,color='red');

"""### Internal Memory vs Price Range"""

sns.pointplot(y="int_memory", x="price_range", data=dataset)

"""### % of Phones which support 3G"""

labels = ["3G-supported",'Not supported']
values=dataset['three_g'].value_counts().values

fig1, ax1 = plt.subplots()
ax1.pie(values, labels=labels, autopct='%1.1f%%',shadow=True,startangle=90)
plt.show()

"""### % of Phones which support 4G

"""

labels4g = ["4G-supported",'Not supported']
values4g = dataset['four_g'].value_counts().values
fig1, ax1 = plt.subplots()
ax1.pie(values4g, labels=labels4g, autopct='%1.1f%%',shadow=True,startangle=90)
plt.show()

"""### Battery power vs Price Range"""

sns.boxplot(x="price_range", y="battery_power", data=dataset)

"""### No of Phones vs Camera megapixels of front and primary camera"""

plt.figure(figsize=(10,6))
dataset['fc'].hist(alpha=0.5,color='blue',label='Front camera')
dataset['pc'].hist(alpha=0.5,color='red',label='Primary camera')
plt.legend()
plt.xlabel('MegaPixels')

"""### Mobile Weght vs Price range"""

sns.jointplot(x='mobile_wt',y='price_range',data=dataset,kind='kde');

"""### Talk time vs Price range"""

sns.pointplot(y="talk_time", x="price_range", data=dataset)

"""## X & Y array"""

X=dataset.drop('price_range',axis=1)

y=dataset['price_range']

"""## Splitting the data"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

"""## Creating & Training Linear Regression Model"""

from sklearn.linear_model import LinearRegression
lm = LinearRegression()

lm.fit(X_train,y_train)

lm.score(X_test,y_test)

"""## Creating & Training KNN Model"""

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)

knn.score(X_test,y_test)

"""### Elbow Method For optimum value of K"""

error_rate = []
for i in range(1,20):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,20),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=5)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

"""## Creating & Training Logistic Regression Model"""

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

logmodel.score(X_test,y_test)

"""## Creating & Training Decision Tree Model"""

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)

dtree.score(X_test,y_test)

"""## Tree Visualization"""

feature_names=['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
       'touch_screen', 'wifi']

#For tree Visualization as kaggle does't support pydotplus just install the pydotplus in your systems's conda terminal
'''
import pydotplus as pydot

from IPython.display import Image

from sklearn.externals.six import StringIO

dot_data = StringIO()

tree.export_graphviz(dtree, out_file=dot_data,feature_names=feature_names)

graph = pydot.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png())'''

#Another way
'''from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=feature_names,filled=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())'''

"""## Creating & Training Random Tree Model"""

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)

rfc.score(X_test,y_test)

"""# Conclusion: KNN & Linear Regression performed the best

## RESULT : Linear Regression
"""

y_pred=lm.predict(X_test)

plt.scatter(y_test,y_pred)

plt.plot(y_test,y_pred)

"""## RESULT: KNN"""

from sklearn.metrics import classification_report,confusion_matrix

pred = knn.predict(X_test)

print(classification_report(y_test,pred))

matrix=confusion_matrix(y_test,pred)
print(matrix)

plt.figure(figsize = (10,7))
sns.heatmap(matrix,annot=True)

"""# Price prediction of Test.csv Using KNN for Prediction

### Import test.csv
"""

data_test=pd.read_csv('D:/test.csv')

data_test.head()

data_test=data_test.drop('id',axis=1)

data_test.head()

"""# Model"""

predicted_price=knn.predict(data_test)

"""# Predicted Price Range"""

predicted_price

"""# Adding Predicted price to test.csv"""

data_test['price_range']=predicted_price

data_test