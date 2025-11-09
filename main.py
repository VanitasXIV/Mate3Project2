import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping # NUEVA: Importamos EarlyStopping
import matplotlib.pyplot as plt

# =======================================================
# PARTE 1: CONFIGURACIÓN Y CARGA DE DATOS
# =======================================================

# Definimos el tamaño de las imágenes
img_height, img_width = 128, 128
batch_size = 64
epochs=45
data_dir = "dataset_billetes"

# 1. Carga del conjunto de entrenamiento
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir + '/train',
    validation_split =0.2, # 20% del entrenamiento para validación
    subset="training",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# 2. Carga del conjunto de VALIDACIÓN
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir + '/train',
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# 3. Carga del conjunto de PRUEBA
test_ds= tf.keras.utils.image_dataset_from_directory(
    data_dir + '/test',
    shuffle=False, # Queremos una evaluación ordenada
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# -------------------------------------------------------
# Obtener class_names antes de .map
# -------------------------------------------------------

# Guardamos los nombres de las clases
class_names = train_ds.class_names
print("Clases detectadas:", class_names)
print("Número de Clases:", len(class_names))
num_classes = len(class_names) # Número de clases elegidas

# 4. Normalización: Escalar los pixeles a [0, 1] y aplicar la capa de re-escalado
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# =======================================================
# PARTE 2: DISEÑO Y ENTRENAMIENTO DE LA CNN (Optimización implementada)
# =======================================================

model = Sequential([
    # Capa 1: Convolución
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    
    Dropout(0.2), # REGULARIZACIÓN 1

    # Capa 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Dropout(0.2), # REGULARIZACIÓN 2

    # Capa 3: Más filtros
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Aplanar y Capas Densas
    Flatten(),
    
    Dense(128, activation='relu'), # Capa Densa Principal
    Dropout(0.5), # REGULARIZACIÓN 3
    
    Dense(num_classes, activation='softmax') # Capa de Salida
])

# Compilación
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Resumen para ver los Parámetros
model.summary()

# -------------------------------------------------------
# IMPLEMENTACIÓN DE EARLY STOPPING
# -------------------------------------------------------

early_stopping = EarlyStopping(
    monitor='val_loss', # Monitorea la pérdida de validación
    patience=10,        # Detiene si no hay mejora en 10 épocas
    mode='min',         # Busca el valor mínimo de val_loss
    restore_best_weights=True # Mantiene los pesos del mejor punto (Época 7-10)
)

# Entrenamiento
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=epochs,
                    callbacks=[early_stopping]) # NUEVO: Aplicamos el callback

# Evaluación en el conjunto de prueba (test)
test_loss, test_acc = model.evaluate(test_ds)
print('\nTest accuracy (Post Early Stopping):', test_acc)

# =======================================================
# PARTE 3: VISUALIZACIÓN DE CURVAS DE APRENDIZAJE
# =======================================================

# Obtiene los datos de entrenamiento y validación del historial
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 4))

# Gráfico de Precisión (Accuracy)
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Precisión de Entrenamiento')
plt.plot(epochs_range, val_acc, label='Precisión de Validación')
plt.legend(loc='lower right')
plt.title('Precisión de Entrenamiento y Validación (Post-Early Stop)')
plt.xlabel('Época')
plt.ylabel('Precisión (Accuracy)')
plt.grid(True)

# Gráfico de Pérdida (Loss)
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Pérdida de Entrenamiento')
plt.plot(epochs_range, val_loss, label='Pérdida de Validación')
plt.legend(loc='upper right')
plt.title('Pérdida de Entrenamiento y Validación (Post-Early Stop)')
plt.xlabel('Época')
plt.ylabel('Pérdida (Loss)')
plt.grid(True)

plt.tight_layout()
plt.show()