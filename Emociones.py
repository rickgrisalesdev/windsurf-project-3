import tensorflow as tf
from tensorflow.keras import layers, models
try:
    import cv2
except ModuleNotFoundError:
    cv2 = None
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

if cv2 is None:
    print(
        "ERROR: Falta OpenCV (cv2). Instalalo con: python -m pip install opencv-python\n"
        "Nota: si estas en servidor/headless, usa: python -m pip install opencv-python-headless"
    )

class ReconocedorEmociones:
    def __init__(self):
        """Inicializa el sistema de reconocimiento de emociones"""
        self.model = None
        self.label_encoder = LabelEncoder()
        self.emociones = ['feliz', 'triste', 'enojado', 'sorprendido', 'neutral', 'miedo', 'asco']
        if cv2 is None:
            raise ModuleNotFoundError(
                "OpenCV (cv2) no esta instalado. Instalalo con: python -m pip install opencv-python"
            )
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.historial_entrenamiento = None
        
    def generar_datos_sinteticos(self, n_muestras=1000):
        """
        Genera datos de entrenamiento sintéticos para demostración
        En un caso real, usaríamos un dataset como FER2013
        """
        print("Generando datos de entrenamiento sintéticos...")
        
        np.random.seed(42)
        datos = []
        etiquetas = []
        
        for emocion_idx, emocion in enumerate(self.emociones):
            for i in range(n_muestras):
                # Generar imagen "facial" sintética (48x48 píxeles)
                # Simulamos diferentes patrones para cada emoción
                
                if emocion == 'feliz':
                    # Sonrisa: más brillante en la parte inferior
                    imagen = np.random.normal(100, 30, (48, 48))
                    imagen[30:40, 10:38] += np.random.normal(50, 10, (10, 28))  # boca
                    imagen[10:20, 15:33] += np.random.normal(20, 5, (10, 18))   # ojos
                    
                elif emocion == 'triste':
                    # Cejas fruncidas, boca hacia abajo
                    imagen = np.random.normal(80, 25, (48, 48))
                    imagen[8:15, 15:33] -= np.random.normal(15, 5, (7, 18))     # cejas
                    imagen[35:42, 15:33] -= np.random.normal(20, 8, (7, 18))    # boca
                    
                elif emocion == 'enojado':
                    # Cejas juntas, ojos entrecerrados
                    imagen = np.random.normal(90, 35, (48, 48))
                    imagen[8:15, 12:36] -= np.random.normal(25, 8, (7, 24))     # cejas
                    imagen[15:25, 15:33] -= np.random.normal(15, 5, (10, 18))   # ojos
                    
                elif emocion == 'sorprendido':
                    # Ojos y boca abiertos
                    imagen = np.random.normal(120, 30, (48, 48))
                    imagen[12:22, 12:36] += np.random.normal(40, 10, (10, 24))  # ojos
                    imagen[30:40, 18:30] += np.random.normal(35, 8, (10, 12))  # boca
                    
                elif emocion == 'neutral':
                    # Expresión neutra
                    imagen = np.random.normal(100, 20, (48, 48))
                    imagen[15:25, 15:33] += np.random.normal(10, 3, (10, 18))   # ojos
                    imagen[35:40, 20:28] += np.random.normal(5, 2, (5, 8))     # boca
                    
                elif emocion == 'miedo':
                    # Ojos abiertos, boca ligeramente abierta
                    imagen = np.random.normal(85, 28, (48, 48))
                    imagen[10:20, 13:35] += np.random.normal(30, 8, (10, 22))   # ojos
                    imagen[32:40, 19:29] += np.random.normal(20, 6, (8, 10))    # boca
                    
                else:  # asco
                    # Nariz arrugada, cejas fruncidas
                    imagen = np.random.normal(95, 32, (48, 48))
                    imagen[20:28, 20:28] -= np.random.normal(20, 6, (8, 8))     # nariz
                    imagen[8:15, 15:33] -= np.random.normal(18, 6, (7, 18))     # cejas
                
                # Normalizar a 0-255
                imagen = np.clip(imagen, 0, 255).astype(np.uint8)
                
                # Aplicar data augmentation
                if np.random.random() > 0.5:
                    imagen = np.fliplr(imagen)  # Flip horizontal
                
                # Añadir ruido
                ruido = np.random.normal(0, 5, (48, 48))
                imagen = np.clip(imagen + ruido, 0, 255).astype(np.uint8)
                
                datos.append(imagen)
                etiquetas.append(emocion)
        
        datos = np.array(datos)
        etiquetas = np.array(etiquetas)
        
        # Codificar etiquetas
        etiquetas_codificadas = self.label_encoder.fit_transform(etiquetas)
        
        print(f"Dataset generado: {len(datos)} imágenes")
        print(f"Distribución de emociones:")
        for emocion in self.emociones:
            count = np.sum(etiquetas == emocion)
            print(f"   {emocion}: {count} imágenes")
        
        return datos, etiquetas_codificadas
    
    def construir_modelo_cnn(self, input_shape=(48, 48, 1)):
        """
        Construye una arquitectura CNN para reconocimiento de emociones
        
        Arquitectura:
        - Bloques convolucionales con BatchNormalization y Dropout
        - Global Average Pooling
        - Capas densas con regularización
        """
        print("Construyendo modelo CNN para reconocimiento de emociones...")
        
        self.model = models.Sequential([
            # Bloque 1
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Bloque 2
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Bloque 3
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Clasificador
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(len(self.emociones), activation='softmax')
        ])
        
        # Compilar modelo
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Mostrar resumen
        self.model.summary()
        print("Modelo CNN construido exitosamente")
    
    def entrenar(self, X_train, y_train, X_val, y_val, epocas=50, batch_size=32):
        """Entrena el modelo CNN"""
        print("Iniciando entrenamiento del modelo CNN...")
        
        # Callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # Data augmentation
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Entrenar
        self.historial_entrenamiento = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epocas,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        print("Entrenamiento completado")
        return self.historial_entrenamiento
    
    def visualizar_entrenamiento(self):
        """Visualiza el progreso del entrenamiento"""
        if self.historial_entrenamiento is None:
            print("No hay historial de entrenamiento disponible")
            return
        
        plt.figure(figsize=(15, 5))
        
        # Accuracy
        plt.subplot(1, 3, 1)
        plt.plot(self.historial_entrenamiento.history['accuracy'], label='Entrenamiento')
        plt.plot(self.historial_entrenamiento.history['val_accuracy'], label='Validación')
        plt.title('Precisión durante entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss
        plt.subplot(1, 3, 2)
        plt.plot(self.historial_entrenamiento.history['loss'], label='Entrenamiento')
        plt.plot(self.historial_entrenamiento.history['val_loss'], label='Validación')
        plt.title('Pérdida durante entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        
        # Learning Rate
        if 'lr' in self.historial_entrenamiento.history:
            plt.subplot(1, 3, 3)
            plt.plot(self.historial_entrenamiento.history['lr'])
            plt.title('Learning Rate')
            plt.xlabel('Época')
            plt.ylabel('LR')
            plt.yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    def predecir_emocion(self, imagen):
        """
        Predice la emoción en una imagen facial
        
        Args:
            imagen: imagen en formato numpy array (48x48)
        """
        if self.model is None:
            print("Modelo no entrenado")
            return None, None
        
        # Preprocesar imagen
        if len(imagen.shape) == 3:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        imagen = cv2.resize(imagen, (48, 48))
        imagen = imagen.astype('float32') / 255.0
        imagen = np.expand_dims(imagen, axis=0)
        imagen = np.expand_dims(imagen, axis=-1)
        
        # Realizar predicción
        prediccion = self.model.predict(imagen, verbose=0)
        clase_predicha = np.argmax(prediccion[0])
        confianza = np.max(prediccion[0])
        
        emocion = self.label_encoder.inverse_transform([clase_predicha])[0]
        
        return emocion, confianza
    
    def detectar_caras_emociones(self, imagen_path):
        """
        Detecta caras y predice emociones en una imagen
        """
        if cv2 is None:
            raise ModuleNotFoundError(
                "OpenCV (cv2) no esta instalado. Instalalo con: python -m pip install opencv-python"
            )
        # Cargar imagen
        imagen = cv2.imread(imagen_path)
        if imagen is None:
            print(f"No se pudo cargar la imagen: {imagen_path}")
            return None
        
        # Convertir a escala de grises
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        # Detectar caras
        caras = self.face_cascade.detectMultiScale(gris, 1.1, 4)
        
        resultados = []
        
        for (x, y, w, h) in caras:
            # Extraer región de interés (ROI)
            roi = gris[y:y+h, x:x+w]
            
            # Predecir emoción
            emocion, confianza = self.predecir_emocion(roi)
            
            if emocion:
                resultados.append({
                    'coordenadas': (x, y, w, h),
                    'emocion': emocion,
                    'confianza': confianza
                })
                
                # Dibujar rectángulo y etiqueta
                cv2.rectangle(imagen, (x, y), (x+w, y+h), (0, 255, 0), 2)
                etiqueta = f"{emocion}: {confianza:.2f}"
                cv2.putText(imagen, etiqueta, (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return imagen, resultados
    
    def evaluar_modelo(self, X_test, y_test):
        """Evalúa el rendimiento del modelo"""
        if self.model is None:
            print("Modelo no entrenado")
            return
        
        # Realizar predicciones
        y_pred_probs = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calcular accuracy
        accuracy = np.mean(y_pred == y_test)
        
        # Matriz de confusión simplificada
        print(f"Evaluación del modelo:")
        print(f"   Accuracy general: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Accuracy por clase
        print("   Accuracy por emoción:")
        for i, emocion in enumerate(self.emociones):
            mask = y_test == i
            if np.sum(mask) > 0:
                acc_clase = np.mean(y_pred[mask] == y_test[mask])
                print(f"     {emocion}: {acc_clase:.4f}")
        
        return accuracy

def demo_reconocimiento_emociones():
    """Función de demostración del sistema de reconocimiento de emociones"""
    print("=" * 60)
    print("DEMOSTRACIÓN: RECONOCIMIENTO FACIAL DE EMOCIONES")
    print("=" * 60)
    
    # Crear instancia
    reconocedor = ReconocedorEmociones()
    
    # Generar datos
    X, y = reconocedor.generar_datos_sinteticos(n_muestras=500)
    
    # Preparar datos para CNN
    X = X.astype('float32') / 255.0
    X = np.expand_dims(X, axis=-1)  # Añadir canal de color
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"División de datos:")
    print(f"   Entrenamiento: {len(X_train)} muestras")
    print(f"   Validación: {len(X_val)} muestras")
    print(f"   Prueba: {len(X_test)} muestras")
    
    # Construir modelo
    reconocedor.construir_modelo_cnn()
    
    # Entrenar
    historial = reconocedor.entrenar(
        X_train, y_train, X_val, y_val, 
        epocas=30, batch_size=32
    )
    
    # Visualizar entrenamiento
    reconocedor.visualizar_entrenamiento()
    
    # Evaluar modelo
    accuracy = reconocedor.evaluar_modelo(X_test, y_test)
    
    # Probar predicciones
    print("\n PRUEBAS DE PREDICCIÓN:")
    print("-" * 40)
    
    # Tomar algunas muestras aleatorias para prueba
    indices_prueba = np.random.choice(len(X_test), 5, replace=False)
    
    for idx in indices_prueba:
        imagen = X_test[idx]
        etiqueta_real = y_test[idx]
        
        emocion_predicha, confianza = reconocedor.predecir_emocion(imagen)
        emocion_real = reconocedor.label_encoder.inverse_transform([etiqueta_real])[0]
        
        print(f"Imagen {idx}:")
        print(f"  Emoción real: {emocion_real}")
        print(f"  Emoción predicha: {emocion_predicha}")
        print(f"  Confianza: {confianza:.4f}")
        print(f"  Correcto: {'(Y)' if emocion_predicha == emocion_real else '(N)'}")
        print()
    
    print("Demostración completada")

if __name__ == "__main__":
    demo_reconocimiento_emociones()
