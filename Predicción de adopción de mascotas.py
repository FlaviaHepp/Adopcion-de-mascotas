"""Descripción general:
El conjunto de datos sobre adopción de mascotas proporciona una visión integral de varios factores que pueden influir en la probabilidad de que 
una mascota sea adoptada en un refugio. Este conjunto de datos incluye información detallada sobre mascotas disponibles para adopción, que cubre 
diversas características y atributos.

Características:
PetID: Identificador único de cada mascota.
PetType: tipo de mascota (p. ej., perro, gato, pájaro, conejo).
Raza: Raza específica de la mascota.
AgeMonths: Edad de la mascota en meses.
Color: Color de la mascota.
Tamaño: Categoría de tamaño de la mascota (Pequeño, Mediano, Grande).
PesoKg: Peso de la mascota en kilogramos.
Vacunado: Estado de vacunación de la mascota (0 - No vacunado, 1 - Vacunado).
HealthCondition: Estado de salud de la mascota (0 - Saludable, 1 - Condición médica).
TimeInShelterDays: Tiempo que la mascota ha estado en el refugio (días).
AdoptionFee: Tarifa de adopción que se cobra por la mascota (en dólares).
Dueño Anterior: Si la mascota tuvo un dueño anterior (0 - No, 1 - Sí).
AdoptionLikelihood: Probabilidad de que la mascota sea adoptada (0 - Improbable, 1 - Probable).
Uso:
Este conjunto de datos es ideal para científicos y analistas de datos interesados ​​en comprender y predecir las tendencias de adopción de mascotas. 
Se puede utilizar para:

Modelado predictivo para determinar la probabilidad de adopción de mascotas.
Analizar el impacto de diversos factores en las tasas de adopción.
Desarrollar estrategias para aumentar las tasas de adopción en refugios.
Advertencia:
Este conjunto de datos se recopiló durante un período de tiempo específico. Por lo tanto, no se puede utilizar para generalizar el comportamiento 
de adopción de mascotas.

Conclusión:
Este conjunto de datos tiene como objetivo respaldar la investigación y las iniciativas centradas en aumentar las tasas de adopción de mascotas 
y garantizar que más mascotas encuentren un hogar definitivo."""

"""
La función apply_models toma características (X) y etiquetas de destino (y) como entrada y realiza las siguientes tareas:

Preprocesamiento de datos:

Divide los datos en conjuntos de entrenamiento y prueba.
Comprueba el desequilibrio de clases y aplica SMOTE (sobremuestreo) si es necesario.
Escala las funciones usando StandardScaler.
Capacitación y evaluación del modelo:

Define un conjunto de modelos de clasificación de aprendizaje automático.
Entrena cada modelo con los datos de entrenamiento.
Evalúa cada modelo según los datos de prueba utilizando precisión y puntuación F1.
Imprime informes detallados (precisión, matriz de confusión, informe de clasificación) para cada modelo.
Aprendizaje conjunto:

Identifica los 3 modelos con mejor rendimiento según la puntuación F1.
Crea dos modelos de conjunto (Clasificador de votación y Clasificador de apilamiento) utilizando los 3 modelos principales.
Evalúa los modelos de conjunto sobre los datos de prueba utilizando precisión, matriz de confusión e informe de clasificación.
En resumen, esta función está diseñada para explorar varios modelos de clasificación, identificar los de mejor rendimiento y potencialmente 
mejorar el rendimiento a través de técnicas de aprendizaje en conjunto..
"""

import numpy as np 
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import seaborn as sns
from scipy.stats import ttest_ind
import urllib.request  # For downloading files
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier, StackingClassifier


# Cargar el conjunto de datos
df = pd.read_csv('pet_adoption_data.csv')

# Mostrar las primeras filas del conjunto de datos
df.head()

# Verifique las dimensiones del conjunto de datos
print(df.shape)

# Resumen estadístico
df.describe()

# Comprobar valores faltantes
print(df.isnull().sum())

# Tipos de datos de columnas
print(df.dtypes)

# Separar características (X) y variable objetivo (y)
X = df.drop(['AdoptionLikelihood', 'PetID'], axis=1)
y = df['AdoptionLikelihood']

# Dividir datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir pasos de preprocesamiento para características numéricas y categóricas
numeric_features = ['AgeMonths', 'WeightKg', 'TimeInShelterDays', 'AdoptionFee']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_features = ['PetType', 'Breed', 'Color', 'Size']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinar pasos de preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Agregar clasificador a la canalización de preprocesamiento
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])

# Ajustar el modelo
clf.fit(X_train, y_train)

# Predicciones
y_pred = clf.predict(X_test)

# Evaluar la precisión
accuracy = accuracy_score(y_test, y_pred)
print(f'Exactitud: {accuracy:.2f}')

# Informe de clasificación
print(classification_report(y_test, y_pred))

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print('Matriz de confusión:')
print(conf_matrix)

# Distribución de la probabilidad de adopción

plt.figure(figsize=(15, 15))
sns.countplot(x='AdoptionLikelihood', data=df)
plt.title('Distribución de la probabilidad de adopción\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Probabilidad de adopción\n')
plt.ylabel('Conteo\n')
plt.show()

# Probabilidad de adopción por tipo y tamaño de mascota
plt.style.use('dark_background')
plt.figure(figsize=(15, 15))
sns.barplot(x='PetType', y='AdoptionLikelihood', hue='Size', data=df)
plt.title('Probabilidad de adopción por tipo y tamaño de mascota\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Tipo de mascota\n')
plt.ylabel('Probabilidad de adopción\n')
plt.legend(title='Tamaño')
plt.show()

# Probabilidad de adopción por tipo y tamaño de mascota
plt.style.use('dark_background')
plt.figure(figsize=(15, 15))
sns.barplot(x='PetType', y='AdoptionLikelihood', hue='Size', data=df)
plt.title('Probabilidad de adopción por tipo y tamaño de mascota\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Tipo de mascota\n')
plt.ylabel('Probabilidad de adopción\n')
plt.legend(title='Tamaño')
plt.show()

# Prueba T para edad, meses y probabilidad de adopción
likely_adopted = df[df['AdoptionLikelihood'] == 1]['AgeMonths']
unlikely_adopted = df[df['AdoptionLikelihood'] == 0]['AgeMonths']

t_stat, p_value = ttest_ind(likely_adopted, unlikely_adopted)
print(f'Estadística de prueba: {t_stat:.2f}, valor p: {p_value:.4f}')

# Diagrama de caja de la tarifa de adopción por probabilidad de adopción
plt.style.use('dark_background')
plt.figure(figsize=(15, 15))
sns.boxplot(x='AdoptionLikelihood', y='AdoptionFee', data=df)
plt.title('Tarifa de adopción por probabilidad de adopción\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Probabilidad de adopción\n')
plt.ylabel('Tarifa de adopción ($)\n')
plt.show()

# Función para ajustar la tarifa de adopción según las características de la mascota
def adjust_adoption_fee(adoption_fee, size, health_condition):
    if health_condition == 1:  # Condición médica
        adoption_fee *= 0.8  # Tarifa con descuento para mascotas con condiciones médicas
    if size == 'Small':
        adoption_fee *= 1.1  # Tarifa ligeramente aumentada para mascotas pequeñas
    return adoption_fee

# Ajustar las tarifas de adopción en el conjunto de datos según el tamaño y el estado de salud
df['AdjustedAdoptionFee'] = df.apply(lambda row: adjust_adoption_fee(row['AdoptionFee'], row['Size'], row['HealthCondition']), axis=1)

# Trazado de distribuciones de tarifas de adopción antes y después del ajuste
plt.style.use('dark_background')
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
sns.histplot(df['AdoptionFee'], bins=20, kde=True, color='plum', edgecolor = "blue")
plt.title('Distribución de la tarifa de adopción original\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Tarifa de adopción ($)\n')
plt.ylabel('Contar\n')

plt.subplot(1, 2, 2)
sns.histplot(df['AdjustedAdoptionFee'], bins=20, kde=True, color='magenta', edgecolor = "white")
plt.title('Distribución ajustada de la tarifa de adopción\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Tarifa de adopción ($)\n')
plt.ylabel('Contar\n')

plt.tight_layout()
plt.show()

# Función para generar descripción de mascota con atributos clave
def generate_pet_description(row):
    description = f"Type: {row['PetType']}\n"
    description += f"Breed: {row['Breed']}\n"
    description += f"Color: {row['Color']}\n"
    description += f"Size: {row['Size']}\n"
    description += f"Age: {row['AgeMonths']} months\n"
    if row['HealthCondition'] == 1:
        description += "Health Condition: Medical condition\n"
    else:
        description += "Health Condition: Healthy\n"
    description += f"Adoption Fee: ${row['AdjustedAdoptionFee']:.2f}\n"
    description += f"Time in Shelter: {row['TimeInShelterDays']} days\n"
    return description

# Agregue una columna de descripción al marco de datos
df['Description'] = df.apply(generate_pet_description, axis=1)

# Ejemplo de evento de adopción virtual que muestra mascotas
def virtual_adoption_event(df, num_pets=5):
    print("¡Bienvenido a nuestro evento de adopción virtual!")
    print("Explora nuestras mascotas destacadas:")
    for i, row in df.sample(num_pets).iterrows():
        print(f"Identificación de mascota: {row['PetID']}")
        print(row['Description'])
        print("------------------------")

# Ejemplo de organización de un evento de adopción virtual
virtual_adoption_event(df)

# Guarde el conjunto de datos ajustado con descripciones para su uso posterior
df.to_csv('adjusted_adoption_data.csv', index=False)
print("Conjunto de datos ajustado con descripciones guardadas correctamente.")

df = pd.read_csv('pet_adoption_data.csv')

#Explorando el marco de datos
df.head()

def get_df_info(df):
    print("\n\033[1mShape of DataFrame:\033[0m ", df.shape)
    print("\n\033[1mColumns in DataFrame:\033[0m ", df.columns.to_list())
    print("\n\033[1mData types of columns:\033[0m\n", df.dtypes)
    
    print("\n\033[1mInformation about DataFrame:\033[0m")
    df.info()
    
    print("\n\033[1mNumber of unique values in each column:\033[0m")
    for col in df.columns:
        print(f"\033[1m{col}\033[0m: {df[col].nunique()}")
        
    print("\n\033[1mNumber of null values in each column:\033[0m\n", df.isnull().sum())
    
    print("\n\033[1mNumber of duplicate rows:\033[0m ", df.duplicated().sum())
    
    print("\n\033[1mDescriptive statistics of DataFrame:\033[0m\n", df.describe().transpose())

# Llame a la función
get_df_info(df)

#Preprocesamiento de datos
# Eliminando la columna 'PetID'
df = df.drop('PetID', axis = 1)
# Divida el marco de datos en características (X) y destino (y)
X = df.drop('AdoptionLikelihood', axis=1)
y = df['AdoptionLikelihood']
# Manejo de variables categóricas en X

X = pd.get_dummies(X)


def apply_models(X, y):
    # Divida los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Comprobar desequilibrio de clases
    class_counts = np.bincount(y_train)
    if len(class_counts) > 2 or np.min(class_counts) / np.max(class_counts) < 0.1:
      print("Desequilibrio de clases detectado. Aplicando SMOTE...")
    
    # Aplicar SMOTE (desequilibrio de clases)
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Inicializar el StandardScaler
    scaler = StandardScaler()

    # Ajuste el escalador a los datos de entrenamiento y transforme tanto los datos de entrenamiento como los de prueba
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Definir los modelos
    models = {
        'LogisticRegression': LogisticRegression(),
        'SVC': SVC(),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(),
        'ExtraTrees': ExtraTreesClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'GradientBoost': GradientBoostingClassifier(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(),
        'CatBoost': CatBoostClassifier(verbose=0)
    }

    # Inicializar un diccionario para contener el rendimiento de cada modelo
    model_performance = {}

    # Aplicar cada modelo
    for model_name, model in models.items():
        print(f"\n\033[1mClassification with {model_name}:\033[0m\n{'-' * 30}")
        
        # Ajustar el modelo a los datos de entrenamiento
        model.fit(X_train, y_train)

        # Hacer predicciones sobre los datos de prueba
        y_pred = model.predict(X_test)

        # Calcule la precisión y la puntuación f1
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Almacenar la interpretación en el diccionario
        model_performance[model_name] = (accuracy, f1)

        # Imprimir la puntuación de precisión
        print("\033[1m**Exactitud**:\033[0m\n", accuracy)

        # Imprime la matriz de confusión
        print("\n\033[1m**Matriz de confusión**:\033[0m\n", confusion_matrix(y_test, y_pred))

        # Imprimir el informe de clasificación
        print("\n\033[1m**Informe de clasificación**:\033[0m\n", classification_report(y_test, y_pred))

    # Ordene los modelos según la puntuación f1 y elija los 3 primeros
    top_3_models = sorted(model_performance.items(), key=lambda x: x[1][1], reverse=True)[:3]
    print("\n\033[1mLos 3 mejores modelos según la puntuación F1:\033[0m\n", top_3_models)

    # Extraiga los nombres de los modelos y los clasificadores de los 3 modelos principales
    top_3_model_names = [model[0] for model in top_3_models]
    top_3_classifiers = [models[model_name] for model_name in top_3_model_names]

    # Crea un Clasificador de Votación con los 3 mejores modelos
    print("\n\033[1mInicializando el clasificador de votación con los 3 mejores modelos...\033[0m\n")
    voting_clf = VotingClassifier(estimators=list(zip(top_3_model_names, top_3_classifiers)), voting='hard')
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)
    print("\n\033[1m**Evaluación del clasificador de votación**:\033[0m\n")
    print("\033[1m**Exactitud**:\033[0m\n", accuracy_score(y_test, y_pred))
    print("\n\033[1m**Matriz de confusión**:\033[0m\n", confusion_matrix(y_test, y_pred))
    print("\n\033[1m**Informe de clasificación**:\033[0m\n", classification_report(y_test, y_pred))

    # Crea un clasificador de apilamiento con los 3 mejores modelos
    print("\n\033[1mInicializando el clasificador de apilamiento con los 3 mejores modelos...\033[0m\n")
    stacking_clf = StackingClassifier(estimators=list(zip(top_3_model_names, top_3_classifiers)))
    stacking_clf.fit(X_train, y_train)
    y_pred = stacking_clf.predict(X_test)
    print("\n\033[1m**Evaluación del clasificador de apilamiento**:\033[0m\n")
    print("\033[1m**Exactitud**:\033[0m\n", accuracy_score(y_test, y_pred))
    print("\n\033[1m**Matriz de confusión**:\033[0m\n", confusion_matrix(y_test, y_pred))
    print("\n\033[1m**Informe de clasificación**:\033[0m\n", classification_report(y_test, y_pred))
# Aplicar la función en X e y
apply_models(X, y)

