"""
PROYECTO: Analítica Predictiva de Egresos Hospitalarios -- INSN-SB
===================================================================
SCRIPT DE BENCHMARKING Y EFICIENCIA ALGORÍTMICA (FASE 2)
Mide el tiempo de ejecución de la Descomposición Espectral (PCA)
"""

import time
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

def benchmark_pca():
    print("=" * 60)
    print("  EJECUTANDO BENCHMARKING DE EFICIENCIA -- NUMPY / SCIPY")
    print("=" * 60)
    
    # Definimos tamaños de muestra simulados (Número de pacientes 'm')
    # Mantenemos las n=15 características vectoriales de la entrega
    m_tamaños = [100, 500, 1000, 2500, 5000, 10000]
    n_features = 15
    tiempos = []
    
    for m in m_tamaños:
        # 1. Generar una matriz R aleatoria estandarizada de tamaño m x n
        np.random.seed(42)
        R_simulada = np.random.randn(m, n_features)
        
        # 2. Medir el tiempo de la Fase 2 (Covarianza + Descomposición Espectral)
        t_inicio = time.time()
        
        # Cálculo de matriz de covarianza C
        C = np.cov(R_simulada.T)
        # Extracción de autovalores y autovectores (Lógica de main.py)
        eigenvalues, eigenvectors = linalg.eigh(C)
        # Ordenamiento espectral
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        
        t_fin = time.time()
        tiempo_total = (t_fin - t_inicio) * 1000 # Convertir a milisegundos
        tiempos.append(tiempo_total)
        
        print(f"  [Size] m = {m:<6} pacientes | Tiempo: {tiempo_total:.4f} ms")
        
    # 3. Graficar los resultados de eficiencia algorítmica
    plt.figure(figsize=(8, 5))
    plt.plot(m_tamaños, tiempos, marker='o', color='#C0392B', lw=2, label='Tiempo medido')
    plt.title('Eficiencia Algorítmica de la Descomposición Espectral (PCA)', fontsize=11, fontweight='bold')
    plt.xlabel('Número de Pacientes Registrados (m)', fontsize=10)
    plt.ylabel('Tiempo de Ejecución (milisegundos)', fontsize=10)
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Guardar gráfica para el informe de GitHub
    plt.savefig("benchmark_eficiencia.png", dpi=150, bbox_inches="tight")
    print("\n  [OK] Gráfica de benchmarking guardada como 'benchmark_eficiencia.png'")
    plt.show()

if __name__ == "__main__":
    benchmark_pca()