import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')

# There is Mr and Mrs in name, lets extract it out.
# Example 'Braund, Mr. Owen Harris'
dataset_title = [i.split(',')[1].split('.')[0].strip() for i in dataset['Name']]
dataset_title = pd.Series(dataset_title)
print(dataset_title.value_counts(sort=True))

dataset['Title'] = dataset_title.replace([['Lady', 'the Countess', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major',
                                           'Rev', 'Sir', 'Jonkheer', 'Dona', 'Ms', 'Mme', 'Mlle'], 'Rare'])

# Whether family was present or not
dataset["FamilyS"] = dataset["SibSp"] + dataset['Parch'] + 1
X_test["FamilyS"] = X_test["SibSp"] + X_test["Parch"] + 1


def family(x):
    if x < 2:
        return 'Single'
    elif x == 2:
        return 'Couple'
    elif x <= 4:
        return 'InterM'
    else:
        return 'Large'


dataset['FamilyS'] = dataset["FamilyS"].apply(family)
X_test['FamilyS'] = X_test["FamilyS"].apply(family)

dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)

X_test['Embarked'].fillna(X_test['Embarked'].mode()[0], inplace=True)
dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
X_test['Age'].fillna(X_test['Age'].median(), inplace=True)
X_test['Fare'].fillna(X_test['Fare'].median(), inplace=True)

dataset = dataset.drop(['PassengerId', 'Cabin', 'Name', 'SibSp', 'Parch', 'Ticket'], axis=1)
X_test_passengers = X_test['PassengerId']
X_test = X_test.drop(['PassengerId', 'Cabin', 'Name', 'SibSp', 'Parch', 'Ticket'], axis=1)

# Features
X_train = dataset.iloc[:, 1:9].values

# Output
Y_train = dataset.iloc[:, 0].values

labelencoder_X_1 = LabelEncoder()
X_train[:, 1] = labelencoder_X_1.fit_transform(X_train[:, 1])
X_train[:, 4] = labelencoder_X_1.fit_transform(X_train[:, 4])
X_train[:, 5] = labelencoder_X_1.fit_transform(X_train[:, 5])
X_train[:, 6] = labelencoder_X_1.fit_transform(X_train[:, 6])


labelencoder_X_2 = LabelEncoder()
X_test[:, 1] = labelencoder_X_2.fit_transform(X_test[:, 1])
X_test[:, 4] = labelencoder_X_2.fit_transform(X_test[:, 4])
X_test[:, 5] = labelencoder_X_2.fit_transform(X_test[:, 5])
X_test[:, 6] = labelencoder_X_2.fit_transform(X_test[:, 6])


one_hot_encoder = OneHotEncoder(categorical_features = [0, 1, 4, 5, 6])
X_train = one_hot_encoder.fit_transform(X_train).toarray()
X_test = one_hot_encoder.fit_transform(X_test).toarray()
