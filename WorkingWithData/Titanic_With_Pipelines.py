import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('train.csv')

df = df.drop(['PassengerId','Name','Ticket','Fare','Cabin'],axis=1)

X = df.drop(["Survived"],axis=1)
Y = df['Survived']

X_train , X_test ,Y_train , Y_test = train_test_split(X,Y, test_size=0.8)

nums_cols = ['Age']
cata_cols = ['Sex','Embarked','Pclass']

num_pipeline = Pipeline([
('imputer',SimpleImputer(strategy='median')),
('scalar', StandardScaler())
])

cata_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline,nums_cols),
    ('cat', cata_pipeline, cata_cols)
])

clf_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', LogisticRegression())
])

clf_pipeline.fit(X_train, Y_train)


print("Model Accuracy:", clf_pipeline.score(X_test, Y_test))





