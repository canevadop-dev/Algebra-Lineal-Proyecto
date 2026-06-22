""" PROYECTO: Análisis de Egresos Hospitalarios — INSN-SB (Abril 2026)
===================================================================
Fase 1: Preparación y limpieza de datos con Pandas
Fase 2: Descomposición Espectral (PCA) con NumPy / SciPy
Fase 3: Proyección Predictiva por Mínimos Cuadrados
Fase 4: Evaluación del Error mediante Norma Euclidiana
"""
 
import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")
 
 
# ══════════════════════════════════════════════════════════════
# FASE 1 — PREPARACIÓN DE DATOS
# ══════════════════════════════════════════════════════════════
 
def cargar_datos_hospital():
    """
    Carga y limpia el dataset de egresos hospitalarios.
    Retorna:
        df   : DataFrame limpio
        R    : Matriz numérica estandarizada (np.ndarray)
        meta : dict con nombres de features y etiquetas
    """
    print("=" * 60)
    print("  FASE 1 — PREPARACIÓN DE DATOS")
    print("=" * 60)
 
    ruta_data = "data/Listado_egresos_hospitalarios_abr_2026.csv"
    df = pd.read_csv(ruta_data, sep=";", encoding="latin1")
 
    print(f"\n  [OK] Data cargada desde: {ruta_data}")
    print(f"       Registros  (m) : {df.shape[0]}")
    print(f"       Variables  (n) : {df.shape[1]}")
    print(f"       Columnas       : {df.columns.tolist()}\n")
 
    # Parsear fechas
    df["FECHA_INGRESO"] = pd.to_datetime(df["FECHA_INGRESO"], dayfirst=True, errors="coerce")
    df["FECHA_EGRESO"]  = pd.to_datetime(df["FECHA_EGRESO"],  dayfirst=True, errors="coerce")
 
    df["ESTANCIA"] = (df["FECHA_EGRESO"] - df["FECHA_INGRESO"]).dt.days
 
    df["EDAD_AÑOS"] = df.apply(
        lambda r: r["EDAD"] if r["TIPO_EDAD"] in ["AÑOS", "YEAR"]
        else (r["EDAD"] / 12.0 if r["TIPO_EDAD"] in ["MESES", "MES"]
              else r["EDAD"] / 365.0), axis=1
    )
 
    df["SEXO_BIN"] = (df["SEXO"] == "MASCULINO").astype(int)
    df["REGION"]   = df["PROCEDENCIA"].str.split("-").str[0]
    df["SERVICIO"] = df["ID"].str.extract(r"([A-Z]+)$")
 
    antes = len(df)
    df = df.dropna(subset=["ESTANCIA", "SERVICIO", "EDAD_AÑOS"])
    print(f"  [OK] Filas eliminadas por nulos: {antes - len(df)}")
    print(f"       Registros válidos          : {len(df)}")
 
    TOP_REG = df["REGION"].value_counts().nlargest(8).index.tolist()
    TOP_SRV = df["SERVICIO"].value_counts().nlargest(10).index.tolist()
 
    df["REGION_CAT"] = df["REGION"].apply(lambda x: x if x in TOP_REG else "OTRO")
    df["SERV_CAT"]   = df["SERVICIO"].apply(lambda x: x if x in TOP_SRV else "OTRO")
 
    reg_dummies  = pd.get_dummies(df["REGION_CAT"], prefix="REG").astype(float)
    serv_dummies = pd.get_dummies(df["SERV_CAT"],   prefix="SRV").astype(float)
 
    X = pd.concat([
        df[["ESTANCIA", "EDAD_AÑOS", "SEXO_BIN"]].reset_index(drop=True),
        reg_dummies.reset_index(drop=True),
        serv_dummies.reset_index(drop=True)
    ], axis=1)
 
    feature_names = X.columns.tolist()
 
    X_arr = X.values.astype(float)
    mu    = X_arr.mean(axis=0)
    sigma = X_arr.std(axis=0)
    sigma[sigma == 0] = 1
    R = (X_arr - mu) / sigma
 
    print(f"\n  [OK] Matriz R generada: {R.shape[0]} x {R.shape[1]}")
    print(f"       Features         : {feature_names}")
    print(f"\n  {'─'*56}")
 
    meta = {
        "feature_names"  : feature_names,
        "region_labels"  : df["REGION_CAT"].values,
        "servicio_labels": df["SERV_CAT"].values,
        "top_regiones"   : TOP_REG,
        "top_servicios"  : TOP_SRV,
        "df"             : df,
        "mu"             : mu,
        "sigma"          : sigma,
    }
 
    return df, R, meta
 
 
# ══════════════════════════════════════════════════════════════
# FASE 2 — DESCOMPOSICIÓN ESPECTRAL (PCA)
# ══════════════════════════════════════════════════════════════
 
def pca_espectral(R, meta):
    """
    Aplica PCA sobre la matriz R usando NumPy / SciPy.
    Retorna dict con autovalores, autovectores y scores.
    """
    print("\n" + "=" * 60)
    print("  FASE 2 — DESCOMPOSICIÓN ESPECTRAL (PCA)")
    print("=" * 60)
 
    feature_names = meta["feature_names"]
 
    C = np.cov(R.T)
    print(f"\n  [OK] Matriz de covarianza: {C.shape}")
 
    eigenvalues, eigenvectors = linalg.eigh(C)
 
    idx          = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
 
    total_var   = eigenvalues.sum()
    var_exp     = eigenvalues / total_var
    cum_var_exp = np.cumsum(var_exp)
 
    scores = R @ eigenvectors
 
    print(f"\n  {'CP':<6} {'Autovalor':>12} {'Var. expl.':>12} {'Var. acum.':>12}")
    print(f"  {'─'*46}")
    for i in range(min(10, len(eigenvalues))):
        marca = " ◄" if cum_var_exp[i] >= 0.70 and (i == 0 or cum_var_exp[i-1] < 0.70) else ""
        print(f"  PC{i+1:<3} {eigenvalues[i]:>12.4f} {var_exp[i]:>11.1%} {cum_var_exp[i]:>11.1%}{marca}")
 
    n_70 = np.searchsorted(cum_var_exp, 0.70) + 1
    print(f"\n  [OK] CPs para explicar >= 70% de varianza: {n_70}")
 
    for pc_idx, pc_name in [(0, "PC1"), (1, "PC2")]:
        loadings = eigenvectors[:, pc_idx]
        order    = np.argsort(np.abs(loadings))[::-1][:6]
        print(f"\n  Loadings dominantes — {pc_name}:")
        for j in order:
            bar = "█" * int(abs(loadings[j]) * 25)
            print(f"    {feature_names[j]:<22} {loadings[j]:+.4f}  {bar}")
 
    print(f"\n  {'─'*56}")
 
    return {
        "eigenvalues"  : eigenvalues,
        "eigenvectors" : eigenvectors,
        "var_exp"      : var_exp,
        "cum_var_exp"  : cum_var_exp,
        "scores"       : scores,
        "n_70"         : n_70,
    }
 
 
# ══════════════════════════════════════════════════════════════
# FASE 3 — PROYECCIÓN PREDICTIVA (MÍNIMOS CUADRADOS)
# ══════════════════════════════════════════════════════════════
 
def proyeccion_minimos_cuadrados(df, pca, meta, n_meses_proyeccion=3):
    """
    Construye una serie temporal mensual de egresos y proyecta
    los siguientes n_meses_proyeccion usando mínimos cuadrados
    (regresión lineal via pseudoinversa de Moore-Penrose).
 
    Sistema:  A @ theta = b
      A     : matriz de diseño (Vandermonde de orden 2)
      theta : coeficientes [a0, a1, a2] (intercepto + tendencia + cuadrático)
      b     : egresos mensuales observados
 
    Retorna dict con datos históricos, coeficientes y proyección.
    """
    print("\n" + "=" * 60)
    print("  FASE 3 — PROYECCIÓN PREDICTIVA (MÍNIMOS CUADRADOS)")
    print("=" * 60)
 
    # ── 3.1 Serie temporal mensual ─────────────────────────────
    df_tmp = df.copy()
    df_tmp["MES"] = df_tmp["FECHA_EGRESO"].dt.to_period("M")
    serie = df_tmp.groupby("MES").size().reset_index(name="EGRESOS")
    serie["t"] = np.arange(len(serie), dtype=float)   # índice temporal
 
    t_obs = serie["t"].values
    y_obs = serie["EGRESOS"].values.astype(float)
 
    print(f"\n  [OK] Serie temporal construida: {len(serie)} periodo(s)")
    print(f"\n  {'Periodo':<12} {'t':>4} {'Egresos':>10}")
    print(f"  {'─'*30}")
    for _, row in serie.iterrows():
        print(f"  {str(row['MES']):<12} {int(row['t']):>4} {int(row['EGRESOS']):>10}")
 
    # ── 3.2 Matriz de diseño (polinomio grado 2) ───────────────
    # A = [1  t  t²]  para cada observación
    grado = min(2, len(t_obs) - 1)   # ajusta si hay pocos periodos
    A = np.column_stack([t_obs**i for i in range(grado + 1)])
 
    print(f"\n  [OK] Matriz de diseño A: {A.shape}  (grado {grado})")
 
    # ── 3.3 Solución por mínimos cuadrados ────────────────────
    # theta = (A^T A)^{-1} A^T b  ≡  pseudoinversa
    theta, residuos, rango, sv = np.linalg.lstsq(A, y_obs, rcond=None)
 
    print(f"  [OK] Coeficientes theta:")
    etiquetas = ["a0 (intercepto)", "a1 (tendencia)", "a2 (cuadrático)"]
    for i, c in enumerate(theta):
        print(f"       {etiquetas[i]:<20} = {c:+.4f}")
 
    # ── 3.4 Valores ajustados (in-sample) ─────────────────────
    y_ajustado = A @ theta
 
    # ── 3.5 Proyección futura ──────────────────────────────────
    t_max    = t_obs[-1]
    t_fut    = np.arange(t_max + 1, t_max + 1 + n_meses_proyeccion, dtype=float)
    A_fut    = np.column_stack([t_fut**i for i in range(grado + 1)])
    y_fut    = A_fut @ theta
 
    # Generar etiquetas de periodos futuros
    ultimo_periodo = serie["MES"].iloc[-1]
    periodos_fut   = [ultimo_periodo + i + 1 for i in range(n_meses_proyeccion)]
 
    print(f"\n  Proyeccion para los proximos {n_meses_proyeccion} mes(es):")
    print(f"  {'Periodo':<12} {'Egresos proyectados':>22}")
    print(f"  {'─'*36}")
    for p, y in zip(periodos_fut, y_fut):
        print(f"  {str(p):<12} {y:>22.1f}")
 
    print(f"\n  {'─'*56}")
 
    return {
        "t_obs"        : t_obs,
        "y_obs"        : y_obs,
        "y_ajustado"   : y_ajustado,
        "t_fut"        : t_fut,
        "y_fut"        : y_fut,
        "theta"        : theta,
        "serie"        : serie,
        "periodos_fut" : periodos_fut,
        "grado"        : grado,
    }
 
 
# ══════════════════════════════════════════════════════════════
# FASE 4 — EVALUACIÓN DEL ERROR (NORMA EUCLIDIANA)
# ══════════════════════════════════════════════════════════════
 
def evaluar_error(proyeccion, pca, R):
    """
    Mide la magnitud del vector de error usando la norma euclidiana.
 
    Error de regresión  : e = y_obs - y_ajustado
    Error de reconstrucción PCA: diferencia entre R original y
                                  la reconstrucción con k componentes.
 
    Retorna dict con métricas de error.
    """
    print("\n" + "=" * 60)
    print("  FASE 4 — EVALUACIÓN DEL ERROR (NORMA EUCLIDIANA)")
    print("=" * 60)
 
    # ── 4.1 Error del modelo de mínimos cuadrados ──────────────
    y_obs      = proyeccion["y_obs"]
    y_ajustado = proyeccion["y_ajustado"]
 
    e_regresion      = y_obs - y_ajustado
    norma_e          = np.linalg.norm(e_regresion)          # ||e||₂
    norma_y          = np.linalg.norm(y_obs)
    error_relativo   = norma_e / norma_y
    rmse             = np.sqrt(np.mean(e_regresion**2))
    mae              = np.mean(np.abs(e_regresion))
 
    print(f"\n  -- Error de Regresion (Minimos Cuadrados) --")
    print(f"  Vector de error e = y_obs - y_ajustado")
    print(f"  {'─'*46}")
    print(f"  Norma euclidiana  ||e||₂  : {norma_e:>12.4f}")
    print(f"  Norma relativa ||e||/||y| : {error_relativo:>12.4f}  ({error_relativo:.1%})")
    print(f"  RMSE                      : {rmse:>12.4f}")
    print(f"  MAE                       : {mae:>12.4f}")
 
    # ── 4.2 Error de reconstrucción PCA ───────────────────────
    eigenvectors = pca["eigenvectors"]
    cum_var_exp  = pca["cum_var_exp"]
    n_70         = pca["n_70"]
 
    print(f"\n  -- Error de Reconstruccion PCA --")
    print(f"  {'k (CPs)':<10} {'Var. acum.':>12} {'||E_rec||₂':>14} {'Error rel.':>12}")
    print(f"  {'─'*52}")
 
    errores_rec = {}
    for k in [n_70, min(n_70 + 2, R.shape[1]), R.shape[1]]:
        V_k      = eigenvectors[:, :k]
        R_rec    = (R @ V_k) @ V_k.T          # reconstrucción con k CPs
        E_rec    = R - R_rec                   # matriz de error
        # Norma de Frobenius = norma euclidiana del vector de errores
        norma_rec = np.linalg.norm(E_rec, "fro")
        norma_R   = np.linalg.norm(R, "fro")
        err_rel   = norma_rec / norma_R
        print(f"  {k:<10} {cum_var_exp[k-1]:>12.1%} {norma_rec:>14.4f} {err_rel:>12.4f}")
        errores_rec[k] = {"norma": norma_rec, "relativa": err_rel}
 
    print(f"\n  [OK] Con {n_70} CPs la norma de error relativa es "
          f"{errores_rec[n_70]['relativa']:.1%}")
    print(f"\n  {'─'*56}")
 
    return {
        "e_regresion"    : e_regresion,
        "norma_e"        : norma_e,
        "error_relativo" : error_relativo,
        "rmse"           : rmse,
        "mae"            : mae,
        "errores_rec"    : errores_rec,
    }
 
 
# ══════════════════════════════════════════════════════════════
# VISUALIZACIONES
# ══════════════════════════════════════════════════════════════
 
COLOR_REG = {
    "LIMA":"#3266AD","ICA":"#E07B39","JUNIN":"#2E9E75",
    "PIURA":"#9B59B6","CAJAMARCA":"#C0392B","ANCASH":"#16A085",
    "SAN MARTIN":"#D4A017","AYACUCHO":"#884EA0","OTRO":"#95A5A6",
}
COLOR_SRV = {
    "ARR":"#3266AD","ERN":"#E07B39","AMO":"#2E9E75","AYA":"#9B59B6",
    "OME":"#C0392B","ARC":"#16A085","ALD":"#D4A017","AMI":"#884EA0",
    "AMA":"#E74C3C","UIS":"#1ABC9C","OTRO":"#95A5A6",
}
 
def graficar(pca, meta, proyeccion, error):
    feature_names   = meta["feature_names"]
    region_labels   = meta["region_labels"]
    servicio_labels = meta["servicio_labels"]
    eigenvalues     = pca["eigenvalues"]
    eigenvectors    = pca["eigenvectors"]
    var_exp         = pca["var_exp"]
    cum_var_exp     = pca["cum_var_exp"]
    scores          = pca["scores"]
 
    fig = plt.figure(figsize=(20, 18))
    fig.suptitle("Egresos Hospitalarios INSN-SB | Abril 2026 — Fases 1-4",
                 fontsize=14, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)
 
    # ── Scree plot (Fase 2) ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    k   = min(15, len(eigenvalues))
    ax1.bar(range(1, k+1), eigenvalues[:k], color="#3266AD", alpha=0.80)
    ax1.axhline(1.0, color="#E07B39", lw=1.5, ls="--", label="Kaiser (λ=1)")
    ax1r = ax1.twinx()
    ax1r.plot(range(1, k+1), cum_var_exp[:k]*100,
              color="#2E9E75", marker="o", ms=5, lw=2, label="Var. acum. %")
    ax1r.axhline(70, color="#2E9E75", lw=1, ls=":", alpha=0.6)
    ax1.set_title("Scree Plot — Autovalores", fontsize=10)
    ax1.set_xlabel("Componente"); ax1.set_ylabel("Autovalor")
    ax1r.set_ylabel("Var. acumulada (%)", color="#2E9E75")
    ax1r.tick_params(axis="y", labelcolor="#2E9E75")
    lines  = ax1.get_legend_handles_labels()
    lines2 = ax1r.get_legend_handles_labels()
    ax1.legend(lines[0]+lines2[0], lines[1]+lines2[1], fontsize=8)
 
    # ── Biplot PC1 vs PC2 — Región ─────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for reg in sorted(set(region_labels)):
        mask = region_labels == reg
        ax2.scatter(scores[mask, 0], scores[mask, 1],
                    c=COLOR_REG.get(reg, "#95A5A6"),
                    label=reg, alpha=0.65, s=28, edgecolors="none")
    ax2.axhline(0, color="gray", lw=0.5, ls="--")
    ax2.axvline(0, color="gray", lw=0.5, ls="--")
    ax2.set_title(f"PC1 vs PC2 — Procedencia\n({var_exp[0]:.1%} + {var_exp[1]:.1%})", fontsize=10)
    ax2.set_xlabel(f"PC1 ({var_exp[0]:.1%})"); ax2.set_ylabel(f"PC2 ({var_exp[1]:.1%})")
    ax2.legend(fontsize=7, ncol=2)
 
    # ── Biplot PC1 vs PC2 — Servicio ───────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    for srv in sorted(set(servicio_labels)):
        mask = servicio_labels == srv
        ax3.scatter(scores[mask, 0], scores[mask, 1],
                    c=COLOR_SRV.get(srv, "#95A5A6"),
                    label=srv, alpha=0.65, s=28, marker="s", edgecolors="none")
    ax3.axhline(0, color="gray", lw=0.5, ls="--")
    ax3.axvline(0, color="gray", lw=0.5, ls="--")
    ax3.set_title(f"PC1 vs PC2 — Servicio\n({var_exp[0]:.1%} + {var_exp[1]:.1%})", fontsize=10)
    ax3.set_xlabel(f"PC1 ({var_exp[0]:.1%})"); ax3.set_ylabel(f"PC2 ({var_exp[1]:.1%})")
    ax3.legend(fontsize=7, ncol=2)
 
    # ── Heatmap de loadings PC1–PC5 ────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    n_pc   = min(5, len(eigenvalues))
    load_m = eigenvectors[:, :n_pc].T
    im     = ax4.imshow(load_m, cmap="RdBu_r", aspect="auto", vmin=-0.6, vmax=0.6)
    ax4.set_xticks(range(len(feature_names)))
    ax4.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
    ax4.set_yticks(range(n_pc))
    ax4.set_yticklabels([f"PC{i+1}" for i in range(n_pc)], fontsize=9)
    ax4.set_title("Heatmap de Loadings — PC1 a PC5", fontsize=10)
    plt.colorbar(im, ax=ax4, fraction=0.02, pad=0.01, label="Loading")
    for i in range(n_pc):
        for j in range(len(feature_names)):
            v = load_m[i, j]
            if abs(v) > 0.15:
                ax4.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=6, color="white" if abs(v) > 0.35 else "black")
 
    # ── Barras loadings PC1 vs PC2 ─────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    top_idx = np.argsort(np.abs(eigenvectors[:, 0]))[::-1][:12]
    y_pos   = np.arange(len(top_idx))
    ax5.barh(y_pos+0.2, eigenvectors[top_idx, 0], height=0.35,
             color="#3266AD", alpha=0.85, label="PC1")
    ax5.barh(y_pos-0.2, eigenvectors[top_idx, 1], height=0.35,
             color="#E07B39", alpha=0.85, label="PC2")
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels([feature_names[i] for i in top_idx], fontsize=8)
    ax5.axvline(0, color="black", lw=0.7)
    ax5.set_title("Top Loadings PC1 y PC2", fontsize=10)
    ax5.set_xlabel("Loading"); ax5.legend(fontsize=9)
 
    # ── Fase 3: Serie temporal + proyección ────────────────────
    ax6 = fig.add_subplot(gs[2, :2])
    t_obs      = proyeccion["t_obs"]
    y_obs      = proyeccion["y_obs"]
    y_ajustado = proyeccion["y_ajustado"]
    t_fut      = proyeccion["t_fut"]
    y_fut      = proyeccion["y_fut"]
    serie      = proyeccion["serie"]
    periodos_fut = proyeccion["periodos_fut"]
 
    etiquetas_obs = [str(p) for p in serie["MES"]]
    etiquetas_fut = [str(p) for p in periodos_fut]
    todas_etiq    = etiquetas_obs + etiquetas_fut
    todos_t       = list(t_obs) + list(t_fut)
 
    ax6.plot(t_obs, y_obs, "o-", color="#3266AD", lw=2, ms=7, label="Egresos observados")
    ax6.plot(t_obs, y_ajustado, "--", color="#2E9E75", lw=2, label="Ajuste MC")
    ax6.plot(t_fut, y_fut, "s--", color="#E07B39", lw=2, ms=8,
             label="Proyeccion futura", zorder=5)
    for tf, yf in zip(t_fut, y_fut):
        ax6.annotate(f"{yf:.0f}", xy=(tf, yf), xytext=(0, 10),
                     textcoords="offset points", ha="center",
                     fontsize=9, color="#E07B39", fontweight="bold")
    ax6.axvline(t_obs[-1] + 0.5, color="gray", lw=1, ls=":", alpha=0.7)
    ax6.set_xticks(todos_t)
    ax6.set_xticklabels(todas_etiq, rotation=30, ha="right", fontsize=9)
    ax6.set_title("Fase 3 — Proyeccion de Egresos Mensuales (Minimos Cuadrados)", fontsize=10)
    ax6.set_xlabel("Periodo"); ax6.set_ylabel("Egresos")
    ax6.legend(fontsize=9); ax6.grid(alpha=0.3)
 
    # ── Fase 4: Error de regresión ─────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    e = error["e_regresion"]
    colores_e = ["#C0392B" if v < 0 else "#3266AD" for v in e]
    ax7.bar(t_obs, e, color=colores_e, alpha=0.80)
    ax7.axhline(0, color="black", lw=0.8)
    ax7.set_xticks(t_obs)
    ax7.set_xticklabels(etiquetas_obs, rotation=30, ha="right", fontsize=9)
    ax7.set_title(
        f"Fase 4 — Vector de Error\n||e||₂ = {error['norma_e']:.2f}  |  "
        f"RMSE = {error['rmse']:.2f}  |  Error rel. = {error['error_relativo']:.1%}",
        fontsize=9
    )
    ax7.set_xlabel("Periodo"); ax7.set_ylabel("Error (obs - ajuste)")
    ax7.grid(alpha=0.3)
 
    plt.savefig("analisis_egresos.png", dpi=150, bbox_inches="tight")
    print("\n  [OK] Figura guardada: analisis_egresos.png")
    plt.show()
 
 
# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
 
if __name__ == "__main__":
 
    # FASE 1 — carga y preprocesamiento → Matriz R
    df, R, meta = cargar_datos_hospital()
 
    # FASE 2 — PCA espectral sobre Matriz R
    pca = pca_espectral(R, meta)
 
    # FASE 3 — proyección por mínimos cuadrados
    proyeccion = proyeccion_minimos_cuadrados(df, pca, meta, n_meses_proyeccion=3)
 
    # FASE 4 — evaluación del error con norma euclidiana
    error = evaluar_error(proyeccion, pca, R)
 
    # Visualizaciones de todas las fases
    graficar(pca, meta, proyeccion, error)
    

