import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.datasets import make_blobs
import warnings
warnings.filterwarnings('ignore')

class SegmentadorClientes:
    def __init__(self):
        self.modelo = None
        self.scaler = StandardScaler()
        self.pca = PCA()
        self.datos_originales = None
        self.datos escalados = None
        self.n_clusters_optimo = None
        
    def generar_datos_clientes(self, n_muestras=300):
        """Genera datos sintéticos de clientes"""
        print("👥 Generando datos sintéticos de clientes...")
        
        np.random.seed(42)
        
        # Crear grupos de clientes con diferentes patrones
        # Grupo 1: Clientes de alto valor (gasto alto, frecuencia alta)
        grupo1 = np.random.multivariate_normal(
            [80, 15, 12, 8], 
            [[15, 5, 3, 2], [5, 8, 2, 1], [3, 2, 6, 1], [2, 1, 1, 3]], 
            n_muestras//3
        )
        
        # Grupo 2: Clientes ocasionales (gasto medio, frecuencia baja)
        grupo2 = np.random.multivariate_normal(
            [35, 3, 8, 4], 
            [[10, 2, 4, 2], [2, 3, 1, 1], [4, 1, 5, 2], [2, 1, 2, 2]], 
            n_muestras//3
        )
        
        # Grupo 3: Clientes nuevos (gasto bajo, frecuencia baja, reciente)
        grupo3 = np.random.multivariate_normal(
            [20, 2, 3, 10], 
            [[8, 2, 2, 3], [2, 2, 1, 2], [2, 1, 3, 2], [3, 2, 2, 4]], 
            n_muestras//3
        )
        
        # Combinar grupos
        X = np.vstack([grupo1, grupo2, grupo3])
        
        # Asegurar valores positivos
        X = np.abs(X)
        
        # Crear DataFrame
        caracteristicas = [
            'Gasto_Mensual',      # Gasto promedio mensual
            'Frecuencia_Compra',  # Compras por mes
            'Antiguedad_Meses',   # Meses como cliente
            'Recencia_Dias'       # Días desde última compra
        ]
        
        df = pd.DataFrame(X, columns=caracteristicas)
        
        print(f"✅ Datos generados: {df.shape[0]} clientes, {df.shape[1]} características")
        print("\n📋 Características:")
        for feat in caracteristicas:
            print(f"   {feat}")
        
        print("\n📈 Estadísticas descriptivas:")
        print(df.describe())
        
        self.datos_originales = df.copy()
        return df
    
    def visualizar_datos_originales(self, df):
        """Visualiza los datos antes del clustering"""
        print("\n📊 Visualizando datos originales...")
        
        # Pair plot
        plt.figure(figsize=(12, 8))
        sns.pairplot(df, diag_kind='hist')
        plt.suptitle('Relaciones entre características - Clientes', y=1.02)
        plt.show()
        
        # Distribuciones individuales
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Distribución de características de clientes', fontsize=16)
        
        for i, col in enumerate(df.columns):
            row, col_idx = i // 2, i % 2
            sns.histplot(df[col], kde=True, ax=axes[row, col_idx])
            axes[row, col_idx].set_title(col)
            axes[row, col_idx].axvline(df[col].mean(), color='red', linestyle='--', label='Media')
            axes[row, col_idx].legend()
        
        plt.tight_layout()
        plt.show()
    
    def preprocesar_datos(self, df):
        """Escala y prepara los datos para clustering"""
        print("\n🔧 Preprocesando datos...")
        
        # Escalar características
        datos_escalados = self.scaler.fit_transform(df)
        self.datos_escalados = pd.DataFrame(datos_escalados, columns=df.columns)
        
        print("✅ Datos escalados (media=0, desviación=1)")
        print("\n📊 Estadísticas después de escalar:")
        print(self.datos_escalados.describe())
        
        return self.datos_escalados
    
    def encontrar_numero_optimo_clusters(self, max_clusters=10):
        """Encuentra el número óptimo de clusters usando múltiples métodos"""
        print("\n🎯 Buscando número óptimo de clusters...")
        
        # Método del codo
        inertias = []
        silhouettes = []
        davies_bouldins = []
        
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.datos_escalados)
            
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(self.datos_escalados, labels))
            davies_bouldins.append(davies_bouldin_score(self.datos_escalados, labels))
        
        # Visualizar métricas
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Método del codo
        axes[0].plot(K_range, inertias, 'bo-')
        axes[0].set_xlabel('Número de clusters (k)')
        axes[0].set_ylabel('Inercia')
        axes[0].set_title('Método del Codo')
        axes[0].grid(True)
        
        # Silhouette
        axes[1].plot(K_range, silhouettes, 'go-')
        axes[1].set_xlabel('Número de clusters (k)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Score')
        axes[1].grid(True)
        
        # Davies-Bouldin
        axes[2].plot(K_range, davies_bouldins, 'ro-')
        axes[2].set_xlabel('Número de clusters (k)')
        axes[2].set_ylabel('Davies-Bouldin Score')
        axes[2].set_title('Davies-Bouldin Score')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Determinar k óptimo (máximo silhouette)
        k_optimo = K_range[np.argmax(silhouettes)]
        self.n_clusters_optimo = k_optimo
        
        print(f"\n🎯 Número óptimo de clusters: {k_optimo}")
        print(f"📊 Silhouette score máximo: {max(silhouettes):.4f}")
        
        return k_optimo, inertias, silhouettes, davies_bouldins
    
    def construir_modelo(self, n_clusters=None):
        """Construye y ajusta el modelo K-means"""
        if n_clusters is None:
            n_clusters = self.n_clusters_optimo
        
        print(f"\n🏗️ Construyendo modelo K-means con {n_clusters} clusters...")
        
        self.modelo = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        
        # Ajustar modelo
        labels = self.modelo.fit_predict(self.datos_escalados)
        
        print("✅ Modelo K-means ajustado")
        print(f"🎯 Inercia final: {self.modelo.inertia_:.4f}")
        
        # Calcular métricas de evaluación
        silhouette_avg = silhouette_score(self.datos_escalados, labels)
        db_score = davies_bouldin_score(self.datos_escalados, labels)
        
        print(f"📊 Silhouette Score: {silhouette_avg:.4f}")
        print(f"📊 Davies-Bouldin Score: {db_score:.4f}")
        
        return labels
    
    def visualizar_clusters(self, labels):
        """Visualiza los clusters formados"""
        print("\n📊 Visualizando clusters...")
        
        # Reducir dimensionalidad para visualización
        pca_2d = PCA(n_components=2)
        datos_pca = pca_2d.fit_transform(self.datos_escalados)
        
        # Crear DataFrame con clusters
        df_visualizacion = pd.DataFrame(datos_pca, columns=['PC1', 'PC2'])
        df_visualizacion['Cluster'] = labels
        
        # Visualización 2D
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.scatterplot(data=df_visualizacion, x='PC1', y='PC2', 
                       hue='Cluster', palette='viridis', s=50, alpha=0.7)
        plt.title('Clusters Visualizados (PCA 2D)')
        plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} varianza)')
        plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} varianza)')
        plt.legend()
        
        # Centroides
        centroides_pca = pca_2d.transform(self.modelo.cluster_centers_)
        plt.scatter(centroides_pca[:, 0], centroides_pca[:, 1], 
                   c='red', s=200, marker='X', label='Centroides')
        plt.legend()
        
        # Visualización con características originales
        plt.subplot(1, 2, 2)
        df_con_clusters = self.datos_originales.copy()
        df_con_clusters['Cluster'] = labels
        
        # Usar las dos características más discriminantes
        feature1, feature2 = 'Gasto_Mensual', 'Frecuencia_Compra'
        sns.scatterplot(data=df_con_clusters, x=feature1, y=feature2,
                       hue='Cluster', palette='viridis', s=50, alpha=0.7)
        plt.title(f'Clusters: {feature1} vs {feature2}')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        
        plt.tight_layout()
        plt.show()
        
        return df_con_clusters
    
    def analizar_clusters(self, df_con_clusters):
        """Analiza las características de cada cluster"""
        print("\n📈 Analizando perfiles de clusters...")
        
        # Estadísticas por cluster
        cluster_stats = df_con_clusters.groupby('Cluster').agg({
            'Gasto_Mensual': ['mean', 'std'],
            'Frecuencia_Compra': ['mean', 'std'],
            'Antiguedad_Meses': ['mean', 'std'],
            'Recencia_Dias': ['mean', 'std'],
            'Cluster': 'count'
        }).round(2)
        
        print("\n📊 Estadísticas por cluster:")
        print(cluster_stats)
        
        # Visualización de perfiles
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Perfiles de Clusters', fontsize=16)
        
        características = ['Gasto_Mensual', 'Frecuencia_Compra', 'Antiguedad_Meses', 'Recencia_Dias']
        
        for i, caracteristica in enumerate(caracteristicas):
            row, col = i // 2, i % 2
            sns.boxplot(data=df_con_clusters, x='Cluster', y=caracteristica, ax=axes[row, col])
            axes[row, col].set_title(f'{caracteristica} por Cluster')
            
        plt.tight_layout()
        plt.show()
        
        # Nombres descriptivos para clusters
        print("\n🏷️ Nombres sugeridos para clusters:")
        
        for cluster_id in range(self.n_clusters_optimo):
            cluster_data = df_con_clusters[df_con_clusters['Cluster'] == cluster_id]
            
            gasto_promedio = cluster_data['Gasto_Mensual'].mean()
            frecuencia_promedio = cluster_data['Frecuencia_Compra'].mean()
            
            if gasto_promedio > 60 and frecuencia_promedio > 10:
                nombre = "Clientes VIP"
                descripcion = "Alto gasto y alta frecuencia"
            elif gasto_promedio > 40 and frecuencia_promedio > 5:
                nombre = "Clientes Leales"
                descripcion = "Gasto moderado y frecuencia regular"
            elif gasto_promedio < 30 and frecuencia_promedio < 5:
                nombre = "Clientes Ocasionales"
                descripcion = "Bajo gasto y baja frecuencia"
            else:
                nombre = f"Cluster {cluster_id}"
                descripcion = "Perfil mixto"
            
            print(f"   Cluster {cluster_id}: {nombre} - {descripcion}")
        
        return cluster_stats
    
    def predecir_cluster(self, nuevo_cliente):
        """Predice el cluster para un nuevo cliente"""
        print("\n🔮 Prediciendo cluster para nuevo cliente...")
        
        # Escalar nuevas características
        cliente_escalado = self.scaler.transform([nuevo_cliente])
        
        # Predecir cluster
        cluster = self.modelo.predict(cliente_escalado)[0]
        
        # Calcular distancia al centroide
        distancia = np.linalg.norm(cliente_escalado - self.modelo.cluster_centers_[cluster])
        
        print(f"🎯 Cluster asignado: {cluster}")
        print(f"📏 Distancia al centroide: {distancia:.4f}")
        
        return cluster, distancia

def demo_clustering_no_supervisado():
    """Demostración completa del ejercicio no supervisado"""
    print("=" * 60)
    print("👥 DEMOSTRACIÓN: CLUSTERING NO SUPERVISADO - CLIENTES")
    print("=" * 60)
    
    # Crear instancia
    segmentador = SegmentadorClientes()
    
    # Generar datos
    df = segmentador.generar_datos_clientes(n_muestras=300)
    
    # Visualizar datos originales
    segmentador.visualizar_datos_originales(df)
    
    # Preprocesar
    df_escalado = segmentador.preprocesar_datos(df)
    
    # Encontrar número óptimo de clusters
    k_optimo, inertias, silhouettes, davies_bouldins = segmentador.encontrar_numero_optimo_clusters()
    
    # Construir y entrenar modelo
    labels = segmentador.construir_modelo(k_optimo)
    
    # Visualizar clusters
    df_con_clusters = segmentador.visualizar_clusters(labels)
    
    # Analizar clusters
    cluster_stats = segmentador.analizar_clusters(df_con_clusters)
    
    # Probar con nuevos clientes
    print("\n🧪 PRUEBAS CON NUEVOS CLIENTES:")
    print("-" * 40)
    
    nuevos_clientes = [
        [85, 18, 15, 5],  # Cliente potencial VIP
        [25, 2, 4, 12],   # Cliente nuevo
        [45, 8, 10, 8],   # Cliente regular
    ]
    
    for i, cliente in enumerate(nuevos_clientes, 1):
        print(f"\nCliente {i}: {cliente}")
        cluster, distancia = segmentador.predecir_cluster(cliente)
    
    print("\n✅ Demostración completada")
    print(f"🎯 Se encontraron {k_optimo} clusters de clientes")

if __name__ == "__main__":
    demo_clustering_no_supervisado()
