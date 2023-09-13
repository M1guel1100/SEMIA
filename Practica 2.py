import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# Cargar el archivo CSV
#data = pd.read_csv('spheres2d10.csv', header=None, names=['feature1', 'feature2', 'feature3', 'label'])
#data = pd.read_csv('spheres2d50.csv', header=None, names=['feature1', 'feature2', 'feature3', 'label'])
data = pd.read_csv('spheres2d70.csv', header=None, names=['feature1', 'feature2', 'feature3', 'label'])

# Definir el porcentaje de datos de entrenamiento y prueba
porcentaje_entrenamiento = 0.8
porcentaje_prueba = 0.2

# Número de particiones
num_particiones = 10

# Definir las características (X) y las etiquetas (y) en el conjunto de datos
X = data[['feature1', 'feature2', 'feature3']]
y = data['label']

# Crear listas para almacenar las puntuaciones de precisión de cada partición
precisiones = []

# Realizar el proceso de particionamiento y clasificación
for _ in range(num_particiones):
    # Dividir el conjunto de datos en entrenamiento y prueba
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=porcentaje_prueba)

    # Crear y entrenar el modelo de perceptrón
    perceptron = Perceptron()
    perceptron.fit(X_entrenamiento, y_entrenamiento)

    # Realizar predicciones en el conjunto de prueba
    y_pred = perceptron.predict(X_prueba)

    # Calcular la precisión y almacenarla en la lista de precisiones
    precision = accuracy_score(y_prueba, y_pred)
    precisiones.append(precision)

    # Crear un gráfico para mostrar los datos y las predicciones
    plt.figure()
    plt.scatter(X_entrenamiento['feature1'], X_entrenamiento['feature2'], c=y_entrenamiento, cmap='viridis')
    plt.title(f'Partición {_ + 1} - Precisión: {precision:.2f}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Calcular la precisión promedio de las particiones
precision_promedio = sum(precisiones) / num_particiones

print(f'Precisión promedio en {num_particiones} particiones: {precision_promedio:.2f}')