import pandas as pd
import numpy as np

#FASE 1: Preparación y carga de datos reales del INSN-SB
def cargar_datos_hospital():
    ruta_data = "data/Listado_egresos_hospitalarios_abr_2026.csv"
    df = pd.read_csv(ruta_data, sep = ";")

    print("¡Data cargada con éxito!")
    print(f"Total de registros de pacientes (m) : {df.shape[0]}")
    print(f"Total de variables detectadas (n): {df.shape[1]}")

    #Extracción de la matriz numerica o codificada para PCA
    R = df.to_numpy()
    return R

if __name__ == "__main__"
    matriz_R = cargar_datos_hospital()
    

