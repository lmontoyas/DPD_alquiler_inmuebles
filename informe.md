# Proyecto Predictivo: Regresión Lineal para precios de inmuebles

**Integrantes:** Diana Ñañez · Leonardo Montoya · Alejandro Marcelo

---

## 1. Análisis exploratorio de datos (EDA): contexto y objetivos

### 1.1 Planteamiento del problema

El mercado inmobiliario es altamente dinámico y el precio de una propiedad está influenciado por múltiples factores: desde su ubicación y tamaño, hasta la presencia de amenidades específicas (piscina, cochera, seguridad, etc.). El objetivo principal del proyecto es construir un modelo de Regresión Lineal capaz de predecir el `Precio` de un inmueble basándose en sus características y entorno.

### 1.2 ¿Qué buscamos al analizar este EDA?

Contamos con un dataset enriquecido de más de 90 variables (numéricas, categóricas y binarias). En esta etapa exploratoria buscamos entender la naturaleza de los datos antes de entrenar cualquier modelo. El análisis se centra en:

- **Calidad de los datos:** identificar y tratar valores nulos, duplicados o inconsistencias.
- **Distribución de las variables:** analizar el comportamiento de la variable objetivo (`Precio`) y las predictoras más importantes (`Area_total`, `Dormitorios`, etc.).
- **Detección de outliers:** encontrar propiedades con precios o características extremadamente raras que distorsionen la regresión.
- **Análisis de correlación:** descubrir qué variables tienen la relación lineal más fuerte con el precio y detectar posibles problemas de multicolinealidad.

### 1.3 ¿Por qué este análisis es fundamental?

Aplica el principio "Garbage In, Garbage Out": si entran datos basura, el modelo arrojará basura. Para que la Regresión Lineal sea válida y precisa, requiere cumplir ciertos supuestos. Este EDA no es solo descriptivo; es una herramienta de diagnóstico que nos permite seleccionar las variables correctas, transformar las que no son normales y limpiar el ruido.

---

## 2. Aplicación del EDA

### 2.1 Calidad de datos

#### Limpieza de nombres de columnas

Se detectaron problemas de codificación en la fuente original (UTF-8 mal interpretado), con nombres como `anx81reas verdes`, `Guardiananxada`, `Jardanxadn`. Se aplicó un mapa de correcciones:

```python
correcciones = {
    'Guardiananxada': 'Guardiania',
    'Jardanxadn': 'Jardin',
    'anx81reas verdes': 'Areas_verdes',
    'Areadeportiva': 'Area_deportiva',
    'Areade BBQ': 'Area_BBQ',
    'Areade sauna': 'Area_sauna',
    'Lavanderanxada': 'Lavanderia',
    'anx81tico': 'Atico',
    'Jardanxadn Interno': 'Jardin_Interno',
    'Guarderanxada': 'Guarderia'
}
```

#### Tipificación inicial

El dataset tiene 97 columnas: 10 numéricas, 86 de texto y 1 booleana. De las 86 columnas tipo `object`, se identificaron cuatro grupos:

| Grupo | N.º | Ejemplos |
|---|---|---|
| Áreas con texto "m2" | 3 | `Descripcion`, `Area_constr`, `Area_total` |
| Binarias (≤ 3 valores únicos) | 69 | `Mascotas`, `Terraza`, `Sotano`, `Patio` |
| Categóricas (≤ 15 valores) | 5 | `Dormitorios`, `Estado de Inmueble`, `Tipo` |
| Texto libre | 9 | `Anunciante`, `Direccion`, `Distrito` |

#### Eliminación de columnas redundantes y conversión de tipos

Se descartaron columnas que duplicaban información o no aportaban (descripciones libres, direcciones, fechas, anunciantes). Los textos `Si`/`No`/`NoEspecifica` se mapearon a `1`/`0`/`NaN` y se forzó a numérico las variables clave:

```python
columnas_a_eliminar = ['Area_total', 'Area_constr', 'Descripcion', 'Direccion',
                       'Anunciante', 'Fecha_pub', 'Ubicacion', 'Balneario', 'hex_id']
df_clean = df.drop(columns=[c for c in columnas_a_eliminar if c in df.columns])
df_clean = df_clean.replace({'Si': 1, 'No': 0, 'NoEspecifica': np.nan})
```

Tras este paso el dataset queda con **88 columnas** (78 `float64`, 6 `object`, 3 `int64`, 1 `bool`).

#### Tratamiento de valores nulos

De las 88 variables, **73 presentan nulos**. Las más críticas:

| Variable | % nulos |
|---|---|
| `Uso_profesional` | 89.0 % |
| `Uso_comercial` | 88.7 % |
| `Luminosidad` | 63.1 % |
| `Centros comerciales cercanos` | 58.6 % |
| `Cerca al mar` | 58.6 % |

Las tres primeras se descartan por nivel insostenible de faltantes. Para el resto se aplican distintas estrategias:

- **Variables internas críticas** (Depósito, Cocina, Lavandería, Balcón, etc.): se eliminan los registros con `NaN` porque la ausencia de información compromete la fiabilidad. Tamaño tras este paso: **(7 440, 88)**.
- **`Area_total_m2`**: se elimina el único nulo → **(7 439, 88)**.
- **`Dormitorios`**: imputación condicionada por la mediana del rango de área (5 cuantiles `Muy Pequeño`–`Muy Grande`). El rango `Muy Pequeño` tiene mediana 3; del `Pequeño` en adelante, mediana 4.
- **Servicios básicos** (`Agua`, `Luz`, `Desague`): se imputan a `1` (asumimos que toda propiedad listada los posee).
- **Amenidades binarias** (Piscina, Gimnasio, Sauna, etc.): se imputan a `0` (la ausencia se interpreta como "no la tiene").
- **Categóricas** (`Estado de Inmueble`, `TipoCochera`): se imputan a `No Especifica`.
- **Variables de entorno** (cercanía al mar, parques, colegios, centros comerciales): se imputan a `0` temporalmente, pendientes de enriquecer con OSM.

#### Tipificación booleana final

Tras la imputación, se identifican automáticamente las columnas que solo contienen `0`/`1` y se convierten a `bool`. Resultado: **70 variables booleanas** (Mascotas, Terraza, Piscina, Gimnasio, Cerca al mar, etc.).

#### Validación de duplicados

Se hicieron dos chequeos:

1. **Duplicados exactos** (todas las columnas iguales): 41 filas eliminadas.
2. **Duplicados aproximados** (mismo `Precio`, `Area_total_m2`, `Dormitorios`, `latitud`, `longitud`): 148 filas eliminadas.

**Tamaño oficial del dataset limpio:** **(7 250, 85)**.

---

### 2.2 Distribución de variables

#### 2.2.1 Variable predicha: `Precio`

La variable objetivo presenta una distribución fuertemente sesgada a la derecha:

| Métrica | Valor |
|---|---|
| Asimetría (skewness) | 83.11 |
| Curtosis | 7 015.99 |

El boxplot revela un punto extremo por encima de los 500 millones, claramente un error de digitación o un caso aislado.

Antes de descartar, se filtra por **precio por metro cuadrado** con un umbral laxo (`Q3 + 3·IQR`) que penaliza solo errores de tipeo evidentes:

```python
df_clean['precio_m2'] = df_clean['Precio'] / df_clean['Area_total_m2']
limite_m2 = q3_m2 + 3.0 * iqr_m2
df_final = df_clean[df_clean['precio_m2'] <= limite_m2].copy()
```

**Transformación logarítmica.** Para estabilizar la varianza y reducir el efecto de los valores extremos, se aplica `log1p`:

| Métrica | Original | Log |
|---|---|---|
| Asimetría | 83.11 | 4.14 |
| Curtosis | 7 015.99 | 33.10 |

**Insights de la variable predicha:**

1. La asimetría y curtosis extremas indican una cola pesada. Es típico en viviendas: la mayoría son de precio medio, pero un puñado de propiedades de lujo distorsiona el promedio. La media (S/. 888 000) está muy alejada de la mediana (S/. 580 000).
2. Tras el log, media (≈ 13.23) y mediana (≈ 13.27) casi coinciden. El análisis de correlación de Pearson recién es válido en la escala log.
3. El boxplot detecta un valor por encima de los 500 millones que el análisis de Mahalanobis confirma como outlier.

#### 2.2.2 Variables predictoras continuas

Para `Area_total_m2`, `Area_constr_m2`, `Antiguedad` y `cantidad_denuncias` se aplica **Yeo-Johnson** solo cuando la asimetría inicial supera 0.75:

```python
pt = PowerTransformer(method='yeo-johnson')
for col in cols_continuas:
    if abs(skew(df_final[col])) > 0.75:
        df_final[f"{col}_Transf"] = pt.fit_transform(df_final[[col]])
```

**Insights:**

1. Las áreas presentan asimetría muy alta antes de transformar: `Area_total_m2` = 4.73, `Area_constr_m2` = 9.35. `Antiguedad` tiene un sesgo moderado (1.38).
2. Yeo-Johnson maneja ceros mejor que un log simple. La asimetría de las áreas baja a ~0.04 y la curtosis de niveles >150 a valores casi nulos.
3. Los histogramas transformados muestran ahora una distribución gausiana, lo cual es requisito ideal para regresión lineal y para que Pearson sea válido.

---

### 2.3 Detección de valores atípicos

#### Análisis univariado (IQR)

Aplicando `Q1 − 1.5·IQR` y `Q3 + 1.5·IQR` sobre `Area_total_m2` se obtiene el rango `[−550, 1450] m²` y se detectan **560 propiedades** con área sospechosa (> 1 450 m²).

**Análisis geográfico.** Esos 560 casos no son necesariamente errores; se concentran en distritos donde existe ese tipo de propiedad:

| Distrito | N.º propiedades > 1 450 m² |
|---|---|
| La Molina | 155 |
| Santiago de Surco | 106 |
| Cieneguilla | 69 |
| Pachacamac | 43 |
| Chorrillos | 27 |
| Chaclacayo | 25 |
| Mala | 12 |
| Asia | 11 |
| Pucusana | 11 |
| Chilca | 8 |

La Molina y Surco concentran mansiones; Cieneguilla, Pachacamac y Mala tienen casas de campo y fundos. Conclusión: muchos "outliers" no son errores sino un segmento distinto del mercado.

#### Análisis multivariado: Distancia de Mahalanobis

Se calcula `D² = (x − μ)ᵀ Σ⁻¹ (x − μ)` sobre las versiones transformadas de `Precio_Log`, `Area_total_m2_Transf` y `Antiguedad_Transf`, comparándola contra el umbral de la distribución χ² con `p = 0.001` y 3 grados de libertad:

```python
cols_mahalanobis = ['Precio_Log', 'Area_total_m2_Transf', 'Antiguedad_Transf']
X = df_final[cols_mahalanobis].values
mu = np.mean(X, axis=0)
inv_cov = np.linalg.inv(np.cov(X.T))
diff = X - mu
df_final['mahalanobis_d2'] = np.sum(diff @ inv_cov * diff, axis=1)

umbral = chi2.ppf(1 - 0.001, df=3)
outliers_multi = df_final[df_final['mahalanobis_d2'] > umbral]
```

**Resultado:** 35 outliers multivariados detectados. Tras eliminarlos, el dataset queda con **7 153 registros**.

**Insights:**

1. La regla del IQR sola identificó 560 casos sospechosos por área, pero el análisis geográfico mostró que muchos corresponden a un segmento legítimo del mercado (campo y mansiones).
2. La frecuencia por antigüedad muestra un pico en propiedades de 1 año, indicando fuerte presencia de proyectos de estreno. La frecuencia cae notablemente a partir de los 5 años: el dataset está sesgado hacia inmuebles modernos.
3. Mahalanobis detecta combinaciones incongruentes (precio muy bajo para área inmensa). Con umbral basado en χ² al 99.9 %, se identifican 35 outliers críticos.

---

### 2.4 Análisis de correlación

#### Matriz de Pearson sobre variables transformadas

Variables analizadas: `Precio_Log`, `Area_total_m2_Transf`, `Area_constr_m2_Transf`, `Dormitorios`, `NroBanios`, `Antiguedad_Transf`, `cantidad_denuncias`.

Hallazgos numéricos clave:

| Relación | r |
|---|---|
| Precio (Log) — Área Total (Transf) | 0.77 |
| Precio (Log) — Área Construida (Transf) | 0.73 |
| Área Total (Transf) — Área Construida (Transf) | 0.70 |

#### Impacto de las amenidades binarias

Para las 70 variables booleanas se calcula la diferencia en la mediana del `Precio_Log` entre `True` y `False`:

```python
resultados = []
for col in df_bool.columns:
    med_true = df_final_limpio[df_final_limpio[col] == True]['Precio_Log'].median()
    med_false = df_final_limpio[df_final_limpio[col] == False]['Precio_Log'].median()
    resultados.append({'Variable': col, 'Incremento_Mediano_Log': med_true - med_false})

ranking_bool = pd.DataFrame(resultados).sort_values('Incremento_Mediano_Log', ascending=False)
```

Las amenidades que más empujan el precio hacia arriba son: piscina, áreas verdes, áreas sociales (BBQ, terraza), seguridad y guardianía.

#### ANOVA por Distrito

```python
distritos_grupos = [g['Precio_Log'].values for _, g in df_final_limpio.groupby('Distrito')]
f_stat, p_val = stats.f_oneway(*distritos_grupos)
```

**Resultado:** F = 42.82, p ≈ 0. El distrito tiene un impacto estadísticamente significativo en el precio.

**Insights:**

1. Existe una correlación positiva muy fuerte entre `Precio (Log)` y las áreas (r = 0.77 para Área Total, 0.73 para Construida). El tamaño es el principal motor de valor en el dataset.
2. Las dos áreas presentan r = 0.70 entre sí, lo cual indica multicolinealidad moderada-alta. Para el modelo conviene conservar una sola o construir un índice combinado.
3. Top amenidades: piscina y áreas verdes encabezan el ranking (segmento de lujo y campo). Seguridad y guardianía actúan como factores de "piso": su ausencia castiga el precio más de lo que su presencia lo eleva. Áreas sociales (BBQ, terraza) tienen un impacto emocional que se traduce en valor de cierre.
4. Con F = 42.82 y p ≈ 0, la ubicación no es secundaria sino un determinante crítico del precio.

---

## 3. Identificación de retos y limitaciones de los datos

- **Calidad de la entrada de datos.** La presencia de outliers extremos (precios de millones) sugiere que las fuentes (portales web) tienen poca validación en el llenado. Por eso fue necesario el filtro por `precio_m2` y la limpieza con Mahalanobis.
- **Sesgo de representatividad.** El dataset está fuertemente inclinado hacia propiedades de ~1 año de antigüedad. Esto limita la capacidad del análisis para entender el mercado de "segundo uso" antiguo o casas coloniales.
- **Codificación de caracteres.** La necesidad de corregir nombres como `anx81reas verdes` indica problemas de UTF-8 en la fuente original, lo que pudo haber causado pérdida de registros durante la integración.
- **Variables vacías ambiguas.** En las booleanas, si un campo no dice "Piscina" no queda claro si la propiedad no la tiene o si simplemente no se llenó el dato. Esta ambigüedad introduce un sesgo difícil de cuantificar.

---

## 4. Reflexión del impacto en el proyecto

Este EDA transformó un dataset sucio y sesgado en una base accionable. Sin las transformaciones de Yeo-Johnson y logaritmo, las conclusiones habrían sido erróneas, guiadas por un puñado de propiedades extremas.

El mayor impacto es la reducción de la incertidumbre:

- El **Área Total** es el predictor más confiable.
- El **Distrito** es una variable obligatoria (p ≈ 0 en ANOVA).
- Se eliminó el ruido de los **35 casos críticos** (Mahalanobis) y los **189 duplicados** (exactos + aproximados) que habrían arruinado cualquier promedio.
- El dataset final de **7 153 registros** queda listo para entrenar la Regresión Lineal con garantías razonables sobre los supuestos del modelo.
