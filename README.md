# Algebra-Lineal-Proyecto Analítica Predictiva de Egresos Hospitalarios - INSN-SB 🏥📊

Este repositorio contiene el desarrollo del proyecto aplicado para el curso de **Álgebra Lineal Aplicada** en la Universidad del Pacífico (2026). El proyecto utiliza datos abiertos del Instituto Nacional de Salud del Niño San Borja (INSNSB) para modelar y predecir flujos de pacientes y optimizar la gestión de recursos críticos en el sector salud mediante herramientas algebraicas avanzadas.

---

## 1. Definición del Problema 📝
El INSNSB maneja un flujo constante y altamente dinámico de pacientes pediátricos de alta complejidad a nivel nacional. La variabilidad mensual en los egresos hospitalarios (por especialidad, diagnóstico, procedencia o grupo etario) dificulta la asignación eficiente de recursos como camas, personal médico y presupuestos operativos. 

El objetivo central es desarrollar un modelo matemático capaz de **predecir la cantidad de egresos hospitalarios futuros** y, en paralelo, **reducir la dimensionalidad de las características clínicas** para identificar patrones ocultos de morbilidad, permitiendo una gestión hospitalaria proactiva basada en datos reales.

---

## 2. Sustento Matemático y Herramientas de Álgebra Lineal 🧮

* **Matrices de Características Clínicas:** Construcción de la matriz de datos $R \in \mathbb{R}^{m \times n}$ a partir de los registros históricos abiertos, donde cada fila representa el perfil vectorial de un paciente egresado.
* **Análisis Espectral y Reducción Dimensional (PCA):** Estudio del espectro de autovalores de la matriz de covarianza de los datos. Actúa de manera análoga a un filtro de frecuencias de Fourier, aislando las tendencias estructurales de salud y eliminando el ruido estadístico al proyectar ortogonalmente los datos en un subespacio de $k$ dimensiones dominantes.
* **Proyecciones Ortogonales y Mínimos Cuadrados ($A^T A \hat{x} = A^T b$):** Formulación de un modelo de regresión lineal multivariable mediante ecuaciones normales. Debido a la inconsistencia natural de los datos reales, se calcula la proyección ortogonal del vector de observaciones sobre el subespacio de variables predictoras temporales para minimizar el error residual.

---

## 3. Plan de Trabajo Inicial (Metodología) 📅

1. **Fase 1 (Preparación de Datos):** Descarga y estructuración de la matriz de datos abiertos de egresos del INSN-SB (Abril 2026). Limpieza y preprocesamiento con `Pandas`.
2. **Fase 2 (Descomposición Espectral):** Implementación de PCA en `Python` (`NumPy`/`SciPy`) para analizar autovalores y agrupar patologías y procedencias dominantes.
3. **Fase 3 (Proyección Predictiva):** Formulación y resolución del sistema por mínimos cuadrados para proyectar las tendencias y demanda hospitalaria de los siguientes meses.
4. **Fase 4 (Evaluación del Error):** Medición de la magnitud del vector de error mediante la norma euclidiana para validar la precisión del modelo frente a datos de prueba.

---

## 4. Bibliografía Base (Format IEEE) 📚
* [1] S. Barnes, E. Hamrock, M. Toerper, et al., "Real-time prediction of hospital length of stay for discharge prioritization," *Journal of the American Medical Informatics Association*, vol. 23, no. e1, pp. e2-e10, Apr. 2016.
* [2] I. T. Jolliffe and J. Cadima, "Principal component analysis: A review and recent developments," *Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences*, vol. 374, no. 2061, p. 20150202, Feb. 2016.
* [3] H. Anton y C. Rorres, *Elementary Linear Algebra: Applications Version*, 11va ed. Hoboken, NJ: Wiley, 2014.
