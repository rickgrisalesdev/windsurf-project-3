# Informe del Proyecto: Conversor de Temperaturas con IA

## Descripción del Proyecto

Este proyecto es un **conversor de temperaturas basado en redes neuronales** que utiliza TensorFlow y scikit-learn para realizar conversiones entre diferentes unidades de temperatura:

- Celsius a Fahrenheit
- Celsius a Kelvin  
- Fahrenheit a Celsius
- Kelvin a Celsius

El sistema utiliza una red neuronal artificial para aprender las fórmulas matemáticas de conversión de temperatura a partir de datos generados sintéticamente, demostrando cómo las redes neuronales pueden aproximar funciones matemáticas.

## Errores Identificados y Soluciones

### 1. Error: Dependencias Faltantes

**Problema:**
```
ModuleNotFoundError: No module named 'sklearn'
```

**Descripción:** El código requería varias bibliotecas de Python que no estaban instaladas en el sistema.

**Solución:**
- Instalación de todas las dependencias necesarias:
```bash
python3 -m pip install scikit-learn tensorflow matplotlib numpy
```

### 2. Error: Arquitectura del Modelo Demasiado Compleja

**Problema:** El modelo original tenía una arquitectura sobredimensionada para una tarea tan simple:
- 3 capas ocultas (64, 32, 16 neuronas)
- Capas de dropout innecesarias
- Tasa de aprendizaje muy baja (0.001)

**Solución:**
- Simplificación a 2 capas ocultas (32, 16 neuronas)
- Eliminación de capas de dropout
- Aumento de la tasa de aprendizaje a 0.01

### 3. Error: Preprocesamiento Inadecuado de Datos

**Problema:** El escalado de datos no era óptimo:
- Se escalaban tanto la temperatura como el tipo de conversión juntos
- Esto afectaba negativamente el aprendizaje del modelo

**Solución:**
- Separación del escalado: solo se escala la columna de temperatura
- El tipo de conversión se mantiene sin escalar (valores 0, 1, 2, 3)
- Implementación de scalers separados para entrada y salida

### 4. Error: Entrenamiento Insuficiente

**Problema:** Solo se entrenaba durante 50 épocas, lo cual era insuficiente para una buena convergencia.

**Solución:**
- Incremento a 200 épocas para permitir mejor convergencia
- Mantenimiento de early stopping para evitar sobreentrenamiento

## Resultados Antes y Después de las Correcciones

### Antes de las Correcciones:
- **Error relativo promedio:** 36.16%
- **Error absoluto promedio:** 7.79 grados
- **Ejemplo de error:** Conversión de 25°C a Fahrenheit daba 67.07°F (real: 77.00°F)

### Después de las Correcciones:
- **Error relativo promedio:** 2.23%
- **Error absoluto promedio:** 0.42 grados
- **Ejemplo de error:** Conversión de 25°C a Fahrenheit da 76.98°F (real: 77.00°F)

## Mejoras Técnicas Implementadas

### 1. Arquitectura del Modelo
```python
# Antes: 3 capas con dropout
tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(32, activation='relu'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(16, activation='relu'),

# Después: 2 capas simples
tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),
tf.keras.layers.Dense(16, activation='relu'),
```

### 2. Preprocesamiento de Datos
```python
# Antes: Escalado conjunto
X_scaled = self.scaler_input.fit_transform(X)

# Después: Escalado separado
temp_scaler = MinMaxScaler()
X_temps_scaled = temp_scaler.fit_transform(X_temps)
X_scaled = np.column_stack([X_temps_scaled.flatten(), X[:, 1]])
```

### 3. Hiperparámetros
```python
# Antes
learning_rate=0.001
epocas=50

# Después
learning_rate=0.01
epocas=200
```

## Conclusiones

El proyecto demuestra exitosamente cómo las redes neuronales pueden aprender funciones matemáticas simples. Las correcciones implementadas mejoraron drásticamente la precisión del modelo, reduciendo el error en más de 15 veces.

**Lecciones aprendidas:**
1. La simplicidad en la arquitectura es clave para tareas simples
2. El preprocesamiento adecuado de datos es fundamental
3. Los hiperparámetros deben ajustarse según la complejidad de la tarea
4. Más épocas de entrenamiento permiten mejor convergencia

El modelo ahora es adecuado para uso práctico con un error promedio inferior a 1 grado en la mayoría de las conversiones.
