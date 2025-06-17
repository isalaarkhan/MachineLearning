import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

data = {
    'Education': ['High School', 'Bachelor', 'Master', 'PhD', 'High School', 'Master']
}

df = pd.DataFrame(data)
print(df)

education_order = ['High School', 'Bachelor', 'Master', 'PhD']

encoder = OrdinalEncoder(categories=[education_order])
print(encoder)

print(encoder.fit_transform(df[['Education']]))