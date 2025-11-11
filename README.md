## Observaciones y Conclusiones del Entrenamiento (CNN Billetes)

El objetivo de los experimentos fue mitigar el **sobreajuste (overfitting)** provocado por el uso de un conjunto de datos limitado de imágenes reales, llevando la precisión de validación a su punto máximo de generalización.

---

### A. Experimento Base y Confirmación del Sobreajuste

**(Análisis de los gráficos de 15 y 45 épocas sin Data Augmentation)**
![](assets\firsttrainingbefore45epochswith15epochs.png)
| Métrica | Observación Clave |
| :--- | :--- |
| **Precisión** | La precisión de entrenamiento ($\approx 90-95\%$) se separa drásticamente de la de validación ($\approx 78\%$) después de la Época 10-15. |
| **Pérdida** | La pérdida de validación deja de mejorar consistentemente alrededor de la Época 10-20, mientras que la pérdida de entrenamiento sigue cayendo. |
| **Conclusión** | Se confirma un **sobreajuste severo**: el modelo memoriza el ruido específico de las imágenes en lugar de aprender características generalizables. |

---

### B. Implementación de Early Stopping  y Data Augmentation

#### 1. Early Stopping

**(Análisis del gráfico `earlyStopping_45epocas.png`)**
![](assets\earlyStopping_45epocas.png)

| Métrica | Observación Clave |
| :--- | :--- |
| **Eficacia** | El *Early Stopping* detiene el entrenamiento automáticamente (en este caso, alrededor de la Época 36). |
| **Función** | Esto evita las épocas innecesarias (Época 36-45) que solo habrían incrementado la pérdida de validación y el sobreajuste. |

#### 2. Data Augmentation (Aumento de Datos)

**(Análisis de los gráficos `DataAugmentation.png` y `finalTest.png`)**
![](assets\DataAugmentation.png)
![](assets\finalTest.png)

| Métrica | Observación Clave |
| :--- | :--- |
| **Estabilidad** | Las curvas de pérdida (entrenamiento y validación) se mantienen **más cercanas** y estables hasta la Época $\mathbf{25}$. |
| **Generalización** | El Aumento de Datos inyectó variabilidad (rotaciones, flips), haciendo que el modelo sea más **robusto** y reduciendo la tasa de sobreajuste inicial. Además nos ayuda a mitigar el principal problema: la falta de volumen en nuestro conjunto de datos|
| **Rendimiento** | La Precisión de Validación alcanza un valor máximo de $\approx 80\% - 85\%$, un límite superior de generalización para la arquitectura actual y los datos base. |

---

### C. Conclusión General del Desarrollo

La arquitectura CNN demostró ser adecuada para el problema:

1.  **La implementación de Data Augmentation y Dropout fue la optimización más efectiva**, permitiendonos entrenar el modelo de forma más estable y mejorando la robustez a variaciones de pose y ángulo.
2.  La gran brecha de precisión ($\approx 10\%-15\%$) que persiste después de la Época 30 indica que el **tamaño y la diversidad del conjunto de datos es el principal factor limitante** para alcanzar una precisión superior al $\mathbf{85\%}$.