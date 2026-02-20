import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import random
from IPython.display import clear_output
import time
import warnings
warnings.filterwarnings('ignore')

class Laberinto:
    def __init__(self, tamaño=6):
        """Crea un laberinto simple"""
        self.tamaño = tamaño
        self.estado_inicial = (0, 0)
        self.estado_objetivo = (tamaño-1, tamaño-1)
        self.obstaculos = self._generar_obstaculos()
        self.estado_actual = self.estado_inicial
        self.terminado = False
        
    def _generar_obstaculos(self):
        """Genera obstáculos aleatorios"""
        obstaculos = set()
        np.random.seed(42)
        
        # Agregar algunos obstáculos aleatorios
        for _ in range(self.tamaño * 2):
            x = np.random.randint(1, self.tamaño-1)
            y = np.random.randint(1, self.tamaño-1)
            if (x, y) != self.estado_objetivo and (x, y) != self.estado_inicial:
                obstaculos.add((x, y))
        
        return obstaculos
    
    def reset(self):
        """Reinicia el laberinto"""
        self.estado_actual = self.estado_inicial
        self.terminado = False
        return self.estado_actual
    
    def paso(self, accion):
        """Ejecuta una acción en el laberinto"""
        x, y = self.estado_actual
        
        # Acciones: 0=arriba, 1=derecha, 2=abajo, 3=izquierda
        if accion == 0:  # arriba
            nuevo_estado = (max(0, x-1), y)
        elif accion == 1:  # derecha
            nuevo_estado = (x, min(self.tamaño-1, y+1))
        elif accion == 2:  # abajo
            nuevo_estado = (min(self.tamaño-1, x+1), y)
        elif accion == 3:  # izquierda
            nuevo_estado = (x, max(0, y-1))
        
        # Verificar si hay obstáculo
        if nuevo_estado in self.obstaculos:
            nuevo_estado = self.estado_actual  # No se mueve
        
        # Calcular recompensa
        if nuevo_estado == self.estado_objetivo:
            recompensa = 100  # Llegó al objetivo
            self.terminado = True
        elif nuevo_estado == self.estado_actual:
            recompensa = -10  # Chocó con obstáculo o pared
        else:
            recompensa = -1   # Paso normal
        
        self.estado_actual = nuevo_estado
        return nuevo_estado, recompensa, self.terminado
    
    def obtener_estado_idx(self, estado):
        """Convierte coordenadas a índice único"""
        return estado[0] * self.tamaño + estado[1]
    
    def visualizar(self, agente_pos=None):
        """Visualiza el laberinto"""
        grid = np.zeros((self.tamaño, self.tamaño))
        
        # Marcar obstáculos
        for obs in self.obstaculos:
            grid[obs] = -1
        
        # Marcar objetivo
        grid[self.estado_objetivo] = 2
        
        # Marcar agente
        if agente_pos is None:
            agente_pos = self.estado_actual
        grid[agente_pos] = 1
        
        # Crear mapa de colores
        plt.figure(figsize=(8, 8))
        sns.heatmap(grid, annot=True, cmap='RdYlGn', center=0,
                   square=True, cbar=False,
                   xticklabels=False, yticklabels=False)
        plt.title('Laberinto (Agente=1, Objetivo=2, Obstáculo=-1)')
        plt.show()

class QLearningAgente:
    def __init__(self, estado_size, accion_size, learning_rate=0.1, descuento=0.95, exploracion=1.0, exploracion_decay=0.995, exploracion_min=0.01):
        """Inicializa el agente Q-Learning"""
        self.estado_size = estado_size
        self.accion_size = accion_size
        self.learning_rate = learning_rate
        self.descuento = descuento
        self.exploracion = exploracion
        self.exploracion_decay = exploracion_decay
        self.exploracion_min = exploracion_min
        
        # Inicializar tabla Q con valores pequeños aleatorios
        self.q_table = np.random.uniform(low=-1, high=1, size=(estado_size, accion_size))
        
        # Historial para visualización
        self.historial_recompensas = []
        self.historial_pasos = []
        self.historial_exploracion = []
        
    def elegir_accion(self, estado):
        """Elige una acción usando política epsilon-greedy"""
        if np.random.random() <= self.exploracion:
            # Exploración: acción aleatoria
            return random.randrange(self.accion_size)
        else:
            # Explotación: mejor acción conocida
            return np.argmax(self.q_table[estado])
    
    def aprender(self, estado, accion, recompensa, siguiente_estado, terminado):
        """Actualiza la tabla Q usando la ecuación de Bellman"""
        # Valor Q actual
        q_actual = self.q_table[estado, accion]
        
        # Mejor valor Q del siguiente estado
        if terminado:
            q_siguiente_max = 0
        else:
            q_siguiente_max = np.max(self.q_table[siguiente_estado])
        
        # Ecuación de Bellman
        q_nuevo = q_actual + self.learning_rate * (recompensa + self.descuento * q_siguiente_max - q_actual)
        
        # Actualizar tabla Q
        self.q_table[estado, accion] = q_nuevo
        
        # Decaimiento de exploración
        if self.exploracion > self.exploracion_min:
            self.exploracion *= self.exploracion_decay
    
    def entrenar_episodio(self, entorno, max_pasos=100):
        """Entrena un episodio completo"""
        estado = entorno.reset()
        estado_idx = entorno.obtener_estado_idx(estado)
        
        recompensa_total = 0
        pasos = 0
        
        for paso in range(max_pasos):
            # Elegir acción
            accion = self.elegir_accion(estado_idx)
            
            # Ejecutar acción
            siguiente_estado, recompensa, terminado = entorno.paso(accion)
            siguiente_estado_idx = entorno.obtener_estado_idx(siguiente_estado)
            
            # Aprender
            self.aprender(estado_idx, accion, recompensa, siguiente_estado_idx, terminado)
            
            # Actualizar estado
            estado = siguiente_estado
            estado_idx = siguiente_estado_idx
            recompensa_total += recompensa
            pasos += 1
            
            if terminado:
                break
        
        return recompensa_total, pasos
    
    def entrenar(self, entorno, episodios=1000, visualizar_cada=100):
        """Entrena el agente durante múltiples episodios"""
        print(f"🚀 Iniciando entrenamiento por {episodios} episodios...")
        
        for episodio in range(episodios):
            recompensa, pasos = self.entrenar_episodio(entorno)
            
            # Guardar historial
            self.historial_recompensas.append(recompensa)
            self.historial_pasos.append(pasos)
            self.historial_exploracion.append(self.exploracion)
            
            # Mostrar progreso
            if (episodio + 1) % visualizar_cada == 0:
                recompensa_promedio = np.mean(self.historial_recompensas[-100:])
                pasos_promedio = np.mean(self.historial_pasos[-100:])
                
                print(f"Episodio {episodio + 1}: "
                      f"Recompensa promedio: {recompensa_promedio:.2f}, "
                      f"Pasos promedio: {pasos_promedio:.1f}, "
                      f"Exploración: {self.exploracion:.3f}")
        
        print("✅ Entrenamiento completado")
    
    def visualizar_entrenamiento(self):
        """Visualiza el progreso del entrenamiento"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Recompensas
        axes[0].plot(self.historial_recompensas, alpha=0.7)
        axes[0].plot(self._suavizar(self.historial_recompensas, 50), 'r-', linewidth=2)
        axes[0].set_title('Recompensa por Episodio')
        axes[0].set_xlabel('Episodio')
        axes[0].set_ylabel('Recompensa')
        axes[0].grid(True)
        
        # Pasos
        axes[1].plot(self.historial_pasos, alpha=0.7)
        axes[1].plot(self._suavizar(self.historial_pasos, 50), 'r-', linewidth=2)
        axes[1].set_title('Pasos hasta el Objetivo')
        axes[1].set_xlabel('Episodio')
        axes[1].set_ylabel('Pasos')
        axes[1].grid(True)
        
        # Tasa de exploración
        axes[2].plot(self.historial_exploracion)
        axes[2].set_title('Tasa de Exploración')
        axes[2].set_xlabel('Episodio')
        axes[2].set_ylabel('ε (epsilon)')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def _suavizar(self, datos, ventana):
        """Suaviza una serie temporal usando media móvil"""
        if len(datos) < ventana:
            return datos
        return np.convolve(datos, np.ones(ventana)/ventana, mode='valid')
    
    def visualizar_politica(self, entorno):
        """Visualiza la política aprendida"""
        print("\n🗺️ Visualizando política aprendida...")
        
        # Crear grid de acciones
        grid_acciones = np.zeros((entorno.tamaño, entorno.tamaño))
        
        for i in range(entorno.tamaño):
            for j in range(entorno.tamaño):
                estado = (i, j)
                if estado not in entorno.obstaculos:
                    estado_idx = entorno.obtener_estado_idx(estado)
                    mejor_accion = np.argmax(self.q_table[estado_idx])
                    grid_acciones[i, j] = mejor_accion
        
        # Crear mapa de calor
        plt.figure(figsize=(10, 8))
        
        # Mostrar laberinto
        grid_laberinto = np.zeros((entorno.tamaño, entorno.tamaño))
        for obs in entorno.obstaculos:
            grid_laberinto[obs] = -1
        grid_laberinto[entorno.estado_objetivo] = 2
        
        # Overlay de política
        sns.heatmap(grid_laberinto, annot=True, cmap='RdYlGn', center=0,
                   square=True, cbar=False,
                   xticklabels=False, yticklabels=False)
        
        # Agregar flechas de política
        acciones_flechas = ['↑', '→', '↓', '←']
        for i in range(entorno.tamaño):
            for j in range(entorno.tamaño):
                if (i, j) not in entorno.obstaculos and (i, j) != entorno.estado_objetivo:
                    estado_idx = entorno.obtener_estado_idx((i, j))
                    mejor_accion = np.argmax(self.q_table[estado_idx])
                    plt.text(j + 0.5, i + 0.5, acciones_flechas[mejor_accion],
                           ha='center', va='center', fontsize=20, fontweight='bold',
                           color='black' if grid_laberinto[i, j] == 0 else 'white')
        
        plt.title('Política Aprendida (Flechas = Mejor Acción)')
        plt.show()
    
    def probar_agente(self, entorno, max_pasos=50, visualizar=True):
        """Prueba al agente entrenado"""
        print("\n🧪 Probando agente entrenado...")
        
        # Desactivar exploración para prueba
        exploracion_original = self.exploracion
        self.exploracion = 0
        
        estado = entorno.reset()
        camino = [estado]
        recompensa_total = 0
        
        for paso in range(max_pasos):
            estado_idx = entorno.obtener_estado_idx(estado)
            accion = self.elegir_accion(estado_idx)
            
            siguiente_estado, recompensa, terminado = entorno.paso(accion)
            camino.append(siguiente_estado)
            recompensa_total += recompensa
            
            if visualizar:
                clear_output(wait=True)
                print(f"Paso {paso + 1}: Acción {['↑', '→', '↓', '←'][accion]}")
                print(f"Estado: {siguiente_estado}, Recompensa: {recompensa}")
                entorno.visualizar(siguiente_estado)
                time.sleep(0.5)
            
            if terminado:
                print(f"🎯 Objetivo alcanzado en {paso + 1} pasos!")
                print(f"🏆 Recompensa total: {recompensa_total}")
                break
        
        # Restaurar exploración
        self.exploracion = exploracion_original
        
        return camino, recompensa_total, terminado

def demo_qlearning_refuerzo():
    """Demostración completa del ejercicio de refuerzo"""
    print("=" * 60)
    print("🤖 DEMOSTRACIÓN: Q-LEARNING - AGENTE EN LABERINTO")
    print("=" * 60)
    
    # Crear entorno
    print("🏗️ Creando laberinto...")
    laberinto = Laberinto(tamaño=6)
    laberinto.visualizar()
    
    # Crear agente
    print("🤖 Creando agente Q-Learning...")
    estado_size = laberinto.tamaño * laberinto.tamaño
    agente = QLearningAgente(
        estado_size=estado_size,
        accion_size=4,  # 4 movimientos posibles
        learning_rate=0.1,
        descuento=0.95,
        exploracion=1.0,
        exploracion_decay=0.995,
        exploracion_min=0.01
    )
    
    # Entrenar agente
    agente.entrenar(laberinto, episodios=1000, visualizar_cada=100)
    
    # Visualizar progreso
    agente.visualizar_entrenamiento()
    
    # Visualizar política aprendida
    agente.visualizar_politica(laberinto)
    
    # Probar agente entrenado
    print("\n🧪 PRUEBA FINAL DEL AGENTE ENTRENADO:")
    print("-" * 40)
    
    # Probar múltiples veces
    for prueba in range(3):
        print(f"\n🎮 Prueba {prueba + 1}:")
        camino, recompensa, exito = agente.probar_agente(laberinto, visualizar=False)
        print(f"   Camino: {len(camino)} pasos")
        print(f"   Éxito: {'✅' if exito else '❌'}")
        print(f"   Recompensa: {recompensa}")
    
    # Mostrar estadísticas finales
    print(f"\n📊 Estadísticas finales:")
    print(f"   Tasa de exploración final: {agente.exploracion:.4f}")
    print(f"   Recompensa promedio últimos 100 episodios: {np.mean(agente.historial_recompensas[-100:]):.2f}")
    print(f"   Pasos promedio últimos 100 episodios: {np.mean(agente.historial_pasos[-100:]):.1f}")
    
    print("\n✅ Demostración completada")

if __name__ == "__main__":
    demo_qlearning_refuerzo()
