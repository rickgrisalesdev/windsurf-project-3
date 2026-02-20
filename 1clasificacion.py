import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class ClasificadorIris:
    def __init__(self):
        self.modelo = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_names = None
        
    def cargar_datos(self):
        """Carga y explora el dataset Iris"""
        print("Cargando dataset Iris...")
        
        # Cargar dataset
        iris = load_iris()
        X = iris.data
        y = iris.target
        
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names
        
        # Crear DataFrame para visualización
        df = pd.DataFrame(X, columns=self.feature_names)
        df['target'] = y
        df['species'] = [self.target_names[i] for i in y]
        
        print(f"Dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
        print(f"Clases: {self.target_names}")
        print(f"Características: {self.feature_names}")
        
        # Mostrar estadísticas
        print("\nEstadísticas descriptivas:")
        print(df.describe())
        
        # Mostrar distribución de clases
        print("\nDistribución de clases:")
        print(df['species'].value_counts())
        
        return X, y, df
    
    def visualizar_datos(self, df):
        """Visualiza el dataset"""
        print("\nGenerando visualizaciones...")
        
        # Pair plot
        plt.figure(figsize=(12, 8))
        sns.pairplot(df, hue='species', palette='viridis')
        plt.suptitle('Relaciones entre características - Dataset Iris', y=1.02)
        plt.show()
        
        # Box plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Distribución de características por especie', fontsize=16)
        
        for i, feature in enumerate(self.feature_names):
            row, col = i // 2, i % 2
            sns.boxplot(data=df, x='species', y=feature, ax=axes[row, col])
            axes[row, col].set_title(feature)
            axes[row, col].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def preprocesar_datos(self, X, y):
        """Divide y preprocesa los datos"""
        print("\nPreprocesando datos...")
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"División de datos:")
        print(f"   Entrenamiento: {X_train.shape[0]} muestras")
        print(f"   Prueba: {X_test.shape[0]} muestras")
        
        # Escalar características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def construir_modelo(self):
        """Construye el modelo de clasificación"""
        print("\nConstruyendo modelo Random Forest...")
        
        self.modelo = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        
        print("Modelo Random Forest construido")
        print(f"Parámetros: {self.modelo.get_params()}")
    
    def entrenar_modelo(self, X_train, y_train):
        """Entrena el modelo"""
        print("\nEntrenando modelo...")
        
        self.modelo.fit(X_train, y_train)
        
        # Importancia de características
        importancia = self.modelo.feature_importances_
        
        print("Modelo entrenado exitosamente")
        print("\nImportancia de características:")
        for nombre, imp in zip(self.feature_names, importancia):
            print(f"   {nombre}: {imp:.4f}")
        
        return importancia
    
    def evaluar_modelo(self, X_test, y_test):
        """Evalúa el rendimiento del modelo"""
        print("\nEvaluando modelo...")
        
        # Predicciones
        y_pred = self.modelo.predict(X_test)
        y_pred_proba = self.modelo.predict_proba(X_test)
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy general: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Reporte de clasificación
        print("\nReporte de clasificación:")
        print(classification_report(y_test, y_pred, 
                                   target_names=self.target_names))
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.target_names,
                   yticklabels=self.target_names)
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicho')
        plt.ylabel('Real')
        plt.show()
        
        return accuracy, cm
    
    def predecir_muestra(self, caracteristicas):
        """Predice una nueva muestra"""
        print("\nRealizando prediccion...")
        
        # Escalar características
        caracteristicas_scaled = self.scaler.transform([caracteristicas])
        
        # Predecir
        prediccion = self.modelo.predict(caracteristicas_scaled)[0]
        probabilidades = self.modelo.predict_proba(caracteristicas_scaled)[0]
        
        print(f"Especie predicha: {self.target_names[prediccion]}")
        print("\nProbabilidades:")
        for i, (especie, prob) in enumerate(zip(self.target_names, probabilidades)):
            print(f"   {especie}: {prob:.4f} ({prob*100:.2f}%)")
        
        return prediccion, probabilidades

def demo_clasificacion_supervisada():
    """Demostración completa del ejercicio supervisado"""
    print("=" * 60)
    print("DEMOSTRACION: CLASIFICACION SUPERVISADA - IRIS")
    print("=" * 60)
    
    # Crear instancia
    clasificador = ClasificadorIris()
    
    # Cargar datos
    X, y, df = clasificador.cargar_datos()
    
    # Visualizar datos
    clasificador.visualizar_datos(df)
    
    # Preprocesar
    X_train, X_test, y_train, y_test = clasificador.preprocesar_datos(X, y)
    
    # Construir modelo
    clasificador.construir_modelo()
    
    # Entrenar
    importancia = clasificador.entrenar_modelo(X_train, y_train)
    
    # Evaluar
    accuracy, cm = clasificador.evaluar_modelo(X_test, y_test)
    
    # Probar con nuevas muestras
    print("\nPRUEBAS CON NUEVAS MUESTRAS:")
    print("-" * 40)
    
    # Muestras de ejemplo
    muestras = [
        [5.1, 3.5, 1.4, 0.2],  # Setosa típica
        [6.7, 3.0, 5.2, 2.3],  # Virginica típica
        [5.9, 3.0, 4.2, 1.5],  # Versicolor típica
    ]
    
    for i, muestra in enumerate(muestras, 1):
        print(f"\nMuestra {i}: {muestra}")
        prediccion, probs = clasificador.predecir_muestra(muestra)
    
    print("\nDemostracion completada")
    print(f"Accuracy final: {accuracy:.4f}")

if __name__ == "__main__":
    demo_clasificacion_supervisada()
