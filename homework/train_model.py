import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

# Cargar el dataset
df = pd.read_csv('files/input/house_data.csv', sep=',')

# Escoger las etiquetas que nos interesan
features = df[
    [
        'bedrooms',
        'bathrooms',
        'sqft_living',
        'sqft_lot',
        'floors',
        'waterfront',
        'condition',
    ]
]

# Escoger la etiqueta a predecir
target = df[['price']]

# Crear el estimador del modelo
estimator = LinearRegression()
estimator.fit(features, target)

# Cargar los datos del modelo a un binario
with open('homework/house_predictor.pkl', 'wb') as file:
    pickle.dump(estimator, file)