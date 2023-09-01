#Importing the required Libaraies
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


#Importing the dataset into our Python file using Pandas
data = pd.read_csv("/Users/mukund/Desktop/MachineLearning/Udemy/TitanicDataset/titanic.csv")

#Dropping the the columns that are not required for making any predicitons 
data.drop(['PassengerId','Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns', inplace=True)

#Splitting the dataset into the target and the data.
target = data.Survived
inputs = data.drop('Survived', axis='columns')

#Preprocessing the data to make it fit for our model by using dummy variables and one hot encoding
dummies = pd.get_dummies(inputs.Sex).astype(int)
inputs = pd.concat([inputs,dummies],axis='columns')
inputs.drop('Sex', axis='columns', inplace=True)
inputs.Age = inputs.Age.fillna(inputs.Age.mean())

#Splitting the dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2, random_state=0)

#Initializing the classifier and fitting our dataset into the model
nb = GaussianNB()
nb.fit(X_train,y_train)

#Making Predicitions
predicitions = nb.predict(X_test)


#Checking the accuracy of the model
def Accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

print("The Accuracy of the model is: ",Accuracy(y_test,predicitions))
