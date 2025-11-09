import tensorflow as tf
from tensorflow import keras

#Definimos el tamaño de las imágenes
img_height, img_width = 128, 128
batch_size = 32
data_dir = "dataset_billetes"

#1. Carga del conjunto de entrenamiento
