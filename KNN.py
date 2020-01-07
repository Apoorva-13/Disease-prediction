import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing the dataset
dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 31].values

# displaying few rows of the dataset
dataset.head()

# there is no missing data 

# taking the count of class
dataset['diagnosis'].value_counts()
#plotting the graph for classes
dataset['diagnosis'].value_counts().plot('bar')


# Get current size
fig_size = plt.rcParams["figure.figsize"]
 
# Prints: [8.0, 6.0]
print ("Current size:", fig_size)
 
# Set figure width to 12 and height to 9
fig_size[0] = 20
fig_size[1] = 15
plt.rcParams["figure.figsize"] = fig_size


#plotting all the columns
dataset.hist()
plt.show()

#Response variable for regression
plt.hist(dataset['radius_mean'], bins = 10 , color = 'red' )


# resizing the plots again for feature plots
fig_size[0] = 15
fig_size[1] = 10
plt.rcParams["figure.figsize"] = fig_size

# feature plots
import seaborn as sns
classy = ['B', 'M']
for x in dataset:
    if(x == 'diagnosis'):
        continue
    for classname in classy:
        subset = dataset[dataset['diagnosis'] == classname]
        #draw the density plot 
        sns.distplot(subset[x],hist = False , kde =True,
                     kde_kws = {'linewidth' : 3},
                     label = classname)
        plt.legend(prop={'size': 16}, title = 'Class')
        plt.title('Density Plot with different classes')
        plt.xlabel(x)
        plt.ylabel('Density') 
    plt.show() 
#feature scaling
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=10)
Y_sklearn = sklearn_pca.fit_transform(X_std)  ## now Y_sklearn has 10 main columns 
explained_variance = sklearn_pca.explained_variance_ratio_
import plotly.plotly as py

import plotly 
plotly.tools.set_credentials_file(username='malvikabhalla99', api_key='CWol1Dp0UR0K9IOh22fs')

import plotly 
plotly.tools.set_config_file(world_readable= True,
                             sharing='public')
data = []
colors = {'B': '#0D76BF', 
          'M': '#00cc96'}
for name, col in zip(('B', 'M'), colors.values()):

    trace = dict(
        type='scatter',
        x=Y_sklearn[y==name,0],
        y=Y_sklearn[y==name,1],
        mode='markers',
        name=name,
        marker=dict(
            color=col,
            size=12,
            line=dict(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5),
            opacity=0.8)
    )
    data.append(trace)

layout = dict(
        xaxis=dict(title='PC1', showline=False),
        yaxis=dict(title='PC2', showline=False)
)
fig = dict(data=data, layout=layout)
plot_url = py.iplot(fig, filename='basic-line')

# encoding the y 
#encoding the categorical data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_Y = LabelEncoder()
y=labelencoder_Y.fit_transform(y) # 1 is for malignant and 0 is for benign

# now for multiple regression our X = Y_sklearn and y = y

X=Y_sklearn
y=y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


# calculating mse
mse = np.mean((y_pred - y_test)**2)
mse

# evaluation using r-square

regressor.score(X_test,y_test)

#residual plot

x_plot = plt.scatter(y_pred, (y_pred - y_test), c='b')

plt.hlines(y=0, xmin= -0.8, xmax=1.8)

plt.title('Residual plot')

from sklearn.linear_model import Ridge

## training the model

ridgeReg = Ridge(alpha=0.05, normalize=True)

ridgeReg.fit(X_train,y_train)

pred = ridgeReg.predict(X_test)

#calculating mse

mse = np.mean((pred - y_test)**2)

mse  
## calculating score
ridgeReg.score(X_test,y_test) 


from yellowbrick.regressor import ResidualsPlot

# Instantiate the linear model and visualizer
ridge = Ridge()
visualizer = ResidualsPlot(ridge)

visualizer.fit(X_train, y_train)  # Fit the training data to the model
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.poof()        
       

##Apply different algos as on X_train,X_test,y_train,y_test

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
pred_y = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print(classification_report(y_test,pred_y))
cm = confusion_matrix(y_test, pred_y)


## find accuracy
from sklearn.metrics import accuracy_score
print ("Accuracy is ", accuracy_score(y_test,pred_y)*100,"% for K-Value:5")

#Confusion Matrix on Heatmap
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title(" KNN Confusion Matrix")
plt.show()


from sklearn import metrics
# Finding the value of k which maximises the accuracy.
def TestBestK(x, z):
    k_range = range(1,100)
    scores = []
    # Accuracy max starts at 0 as there is no risk an accuracy will be below 0
    maxAcc = 0
    # Testing the accuracy for k between 1 and 29
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors = k, metric = 'minkowski', p = 2)
        knn.fit(x, z)
        prediction = knn.predict(X_test)
        scores.append(metrics.accuracy_score(y_test,prediction))
        if metrics.accuracy_score(y_test,prediction) > maxAcc:
            maxAcc = metrics.accuracy_score(y_test,prediction)
            bestNb = k #bestNb is the k which maximises the accuracy (no cross-validation)
        else:
            pass
    # Plot the graph : values of k in the x-axis, acccuracy on the y-axis
    plt.figure()
    plt.title('Accuracy using a test set \n The best accuracy is reached with k =%i' %bestNb)
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.scatter(k_range, scores)
    plt.xticks([0,5,10,20,30,40,50,60,70,80,90,100]);
    return bestNb

TestBestK(X_train,y_train)

knn=KNeighborsClassifier(n_neighbors=5)
# for display purposes, we fit the model on the first two components i.e. PC1, and PC2
knn.fit(X_train[:,0:2], y_train)
# Plotting the decision boundary for all data (both train and test)
# Create color maps
from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#AAFFAA','#FFAAAA'])
cmap_bold = ListedColormap(['#0000FF','#FF0000'])
# creating a meshgrid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h=0.05
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
xy_mesh=np.c_[xx.ravel(), yy.ravel()]
Z = knn.predict(xy_mesh)

Z = Z.reshape(xx.shape)
#print(Z)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
ax=plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max());plt.ylim(yy.min(), yy.max())
plt.xlabel('PC1');plt.ylabel('PC2')
plt.title('KNN')
plt.show()
