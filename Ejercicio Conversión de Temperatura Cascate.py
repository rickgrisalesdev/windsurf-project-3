"""
CONVERSOR DE TEMPERATURAS USANDO TENSORFLOW
============================================
Este modelo demuestra cómo usar una red neuronal para aprender conversiones
entre diferentes unidades de temperatura (Celsius, Fahrenheit, Kelvin).

CONCEPTOS CLAVE:
- Redes neuronales simples (perceptrón)
- Normalización de datos
- Entrenamiento supervisado
- Funciones de activación
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class ConversorTemperatura:
    def __init__(self):
        """Inicializa el conversor con el modelo de red neuronal"""
        self.model = None
        self.scaler_input = MinMaxScaler()
        self.scaler_output = MinMaxScaler()
        self.historial_entrenamiento = None
        
    def preparar_datos(self, n_muestras=1000):
        """
        Genera datos de entrenamiento para conversiones de temperatura
        
        Fórmulas de conversión:
        - F = C × 9/5 + 32
        - K = C + 273.15
        - C = (F - 32) × 5/9
        - C = K - 273.15
        """
        print("📊 Preparando datos de entrenamiento...")
        
        # Generar temperaturas aleatorias en diferentes rangos
        np.random.seed(42)
        
        # Temperaturas Celsius (-50 a 150)
        celsius = np.random.uniform(-50, 150, n_muestras)
        
        # Convertir a otras unidades
        fahrenheit = celsius * 9/5 + 32
        kelvin = celsius + 273.15
        
        # Crear dataset con diferentes conversiones
        datos = []
        
        # Celsius a Fahrenheit
        for i in range(n_muestras):
            datos.append([celsius[i], 0, fahrenheit[i]])  # 0 = Celsius a Fahrenheit
            
        # Celsius a Kelvin
        for i in range(n_muestras):
            datos.append([celsius[i], 1, kelvin[i]])    # 1 = Celsius a Kelvin
            
        # Fahrenheit a Celsius
        for i in range(n_muestras):
            datos.append([fahrenheit[i], 2, celsius[i]])  # 2 = Fahrenheit a Celsius
            
        # Kelvin a Celsius
        for i in range(n_muestras):
            datos.append([kelvin[i], 3, celsius[i]])     # 3 = Kelvin a Celsius
            
        datos = np.array(datos)
        
        # Separar características y etiquetas
        X = datos[:, :2]  # [temperatura, tipo_conversión]
        y = datos[:, 2:3]  # temperatura_resultante
        
        # Normalizar datos
        X_scaled = self.scaler_input.fit_transform(X)
        y_scaled = self.scaler_output.fit_transform(y)
        
        print(f"✅ Dataset creado: {len(datos)} muestras")
        print(f"📈 Rango de temperaturas: {X[:, 0].min():.1f}° a {X[:, 0].max():.1f}°")
        
        return X_scaled, y_scaled, X, y
    
    def construir_modelo(self):
        """
        Construye la arquitectura de la red neuronal
        
        Arquitectura:
        - Capa de entrada: 2 neuronas (temperatura, tipo_conversión)
        - Capa oculta 1: 64 neuronas con activación ReLU
        - Capa oculta 2: 32 neuronas con activación ReLU
        - Capa de salida: 1 neurona (temperatura convertida)
        """
        print("🏗️ Construyendo modelo de red neuronal...")
        
        self.model = tf.keras.Sequential([
            # Capa de entrada
            tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
            tf.keras.layers.Dropout(0.2),
            
            # Capa oculta 1
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            
            # Capa oculta 2
            tf.keras.layers.Dense(16, activation='relu'),
            
            # Capa de salida
            tf.keras.layers.Dense(1, activation='linear')
        ])
        
        # Compilar modelo
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',  # Error cuadrático medio
            metrics=['mae']  # Error absoluto medio
        )
        
        # Mostrar resumen del modelo
        self.model.summary()
        print("✅ Modelo construido exitosamente")
    
    def entrenar(self, X_train, y_train, epocas=100, batch_size=32):
        """
        Entrena el modelo con los datos preparados
        """
        print("🚀 Iniciando entrenamiento...")
        
        # Callback para early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Entrenar modelo
        self.historial_entrenamiento = self.model.fit(
            X_train, y_train,
            epochs=epocas,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        print("✅ Entrenamiento completado")
        return self.historial_entrenamiento
    
    def visualizar_entrenamiento(self):
        """Visualiza el progreso del entrenamiento"""
        if self.historial_entrenamiento is None:
            print("❌ No hay historial de entrenamiento disponible")
            return
        
        plt.figure(figsize=(12, 4))
        
        # Gráfico de pérdida
        plt.subplot(1, 2, 1)
        plt.plot(self.historial_entrenamiento.history['loss'], label='Entrenamiento')
        plt.plot(self.historial_entrenamiento.history['val_loss'], label='Validación')
        plt.title('Pérdida durante entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Pérdida (MSE)')
        plt.legend()
        
        # Gráfico de error absoluto medio
        plt.subplot(1, 2, 2)
        plt.plot(self.historial_entrenamiento.history['mae'], label='Entrenamiento')
        plt.plot(self.historial_entrenamiento.history['val_mae'], label='Validación')
        plt.title('Error Absoluto Medio')
        plt.xlabel('Época')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def convertir_temperatura(self, temperatura, tipo_conversion):
        """
        Realiza conversión de temperatura usando el modelo entrenado
        
        Args:
            temperatura: valor numérico de la temperatura
            tipo_conversion: 
                0 = Celsius a Fahrenheit
                1 = Celsius a Kelvin
                2 = Fahrenheit a Celsius
                3 = Kelvin a Celsius
        """
        if self.model is None:
            print("❌ Modelo no entrenado. Primero ejecuta entrenar()")
            return None
        
        # Preparar entrada
        entrada = np.array([[temperatura, tipo_conversion]])
        entrada_scaled = self.scaler_input.transform(entrada)
        
        # Realizar predicción
        prediccion_scaled = self.model.predict(entrada_scaled)
        prediccion = self.scaler_output.inverse_transform(prediccion_scaled)
        
        return prediccion[0][0]
    
    def conversion_tradicional(self, temperatura, tipo_conversion):
        """Método tradicional para comparar resultados"""
        if tipo_conversion == 0:  # Celsius a Fahrenheit
            return temperatura * 9/5 + 32
        elif tipo_conversion == 1:  # Celsius a Kelvin
            return temperatura + 273.15
        elif tipo_conversion == 2:  # Fahrenheit a Celsius
            return (temperatura - 32) * 5/9
        elif tipo_conversion == 3:  # Kelvin a Celsius
            return temperatura - 273.15
    
    def evaluar_modelo(self, X_test, y_test):
        """Evalúa la precisión del modelo"""
        if self.model is None:
            print("❌ Modelo no entrenado")
            return
        
        # Realizar predicciones
        y_pred_scaled = self.model.predict(X_test)
        y_pred = self.scaler_output.inverse_transform(y_pred_scaled)
        y_true = self.scaler_output.inverse_transform(y_test)
        
        # Calcular error promedio
        error_promedio = np.mean(np.abs(y_pred - y_true))
        error_relativo = np.mean(np.abs((y_pred - y_true) / y_true)) * 100
        
        print(f"📊 Evaluación del modelo:")
        print(f"   Error absoluto promedio: {error_promedio:.4f}°")
        print(f"   Error relativo promedio: {error_relativo:.2f}%")
        
        return error_promedio, error_relativo

def demo_conversor_temperatura():
    """Función de demostración del conversor de temperatura"""
    print("=" * 60)
    print("🌡️  DEMOSTRACIÓN: CONVERSOR DE TEMPERATURAS CON IA")
    print("=" * 60)
    
    # Crear instancia del conversor
    conversor = ConversorTemperatura()
    
    # Preparar datos
    X, y, X_raw, y_raw = conversor.preparar_datos(n_muestras=2000)
    
    # Construir modelo
    conversor.construir_modelo()
    
    # Entrenar modelo
    historial = conversor.entrenar(X, y, epocas=50)
    
    # Visualizar entrenamiento
    conversor.visualizar_entrenamiento()
    
    # Evaluar modelo
    conversor.evaluar_modelo(X, y)
    
    # Probar conversiones
    print("\n🧪 PRUEBAS DE CONVERSIÓN:")
    print("-" * 40)
    
    pruebas = [
        (25, 0, "Celsius a Fahrenheit"),
        (100, 1, "Celsius a Kelvin"),
        (32, 2, "Fahrenheit a Celsius"),
        (273.15, 3, "Kelvin a Celsius"),
        (0, 0, "Celsius a Fahrenheit"),
        (-40, 0, "Celsius a Fahrenheit")
    ]
    
    for temp, tipo, descripcion in pruebas:
        resultado_ia = conversor.convertir_temperatura(temp, tipo)
        resultado_real = conversor.conversion_tradicional(temp, tipo)
        error = abs(resultado_ia - resultado_real)
        
        print(f"{descripcion}:")
        print(f"  Entrada: {temp:.2f}")
        print(f"  Resultado IA: {resultado_ia:.2f}")
        print(f"  Resultado Real: {resultado_real:.2f}")
        print(f"  Error: {error:.4f}")
        print()
    
    print("✅ Demostración completada")

if __name__ == "__main__":
    demo_conversor_temperatura()
