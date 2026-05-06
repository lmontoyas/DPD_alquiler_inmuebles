# LLAVE EN MANO
## Primer Entregable — DS3022 Desarrollo de Producto de Datos
**Universidad de Ingeniería y Tecnología (UTEC)**  
**Equipo:** Leonardo Montoya · Diana Ñañez · Alejandro Marcelo  
**Fecha:** Abril 2026

---

## 1. Definición del Problema y Tema Elegido

El mercado de alquiler de departamentos en Lima Metropolitana presenta una marcada **asimetría de información**: quienes buscan arrendar no cuentan con herramientas objetivas para determinar si el precio publicado en un portal refleja el valor real del inmueble.

El problema puede formularse de la siguiente manera:

> **¿Cómo predecir el precio justo de alquiler de un departamento en Lima, integrando las características físicas del inmueble, datos de seguridad del entorno y la disponibilidad de servicios cercanos?**

Esta pregunta tiene tres dimensiones claras:

1. **Caracterización del inmueble**: área, número de dormitorios y baños, piso, estacionamiento, estado de amoblamiento, antigüedad del edificio.
2. **Seguridad del entorno**: incidencia delictiva por distrito y zona, según datos de la Policía Nacional del Perú (PNP) e INEI.
3. **Servicios y equipamiento urbano**: proximidad a supermercados, centros de salud, colegios, parques, paraderos de transporte y centros comerciales, obtenida vía Google Maps API y OpenStreetMap.

El producto resultante, denominado **LLAVE EN MANO**, es una plataforma interactiva que permite al usuario ingresar las características de un departamento y obtener: (a) el precio estimado de alquiler, (b) un score de calidad del entorno, y (c) un indicador de si el precio publicado es justo, sobrevaluado o subvaluado respecto al mercado.

---

## 2. Relevancia y Justificación del Proyecto

### ¿Por qué es importante este problema?

Lima es una de las ciudades con mayor dinamismo inmobiliario de América del Sur. Según el INEI, más del 30 % de hogares en Lima Metropolitana habita en vivienda alquilada. Sin embargo, la fijación de precios de alquiler opera de forma opaca: los portales como Urbania y Adondevivir listan precios sin contexto comparativo, lo que pone al arrendatario en desventaja frente al propietario y al agente inmobiliario.

Esto genera consecuencias directas:
- Los arrendatarios **pagan en exceso** por departamentos en zonas con alta inseguridad o sin servicios básicos cercanos.
- La búsqueda de departamento se extiende innecesariamente por falta de referencia objetiva de precio.
- No existe un servicio público ni privado accesible que consolide estas tres dimensiones (inmueble, seguridad, entorno) en una sola predicción de precio.

### ¿Por qué resolverlo con datos?

El problema es inherentemente modelable con datos:
- **Los precios siguen patrones estructurados** según distrito, zona, características físicas y entorno urbano.
- **Los datos existen y son accesibles**: portales inmobiliarios públicos (scraping), datos policiales (PNP/INEI), datos geoespaciales (Google Maps API, OSM, censos INEI).
- **El enfoque de machine learning** (modelos de regresión tipo XGBoost o LightGBM) ha demostrado alta eficacia en tareas de predicción de precios inmobiliarios en contextos similares (Zillow, Properati Argentina).

### Valor del proyecto

| Dimensión | Valor generado |
|-----------|----------------|
| Para el arrendatario | Mayor poder de negociación y decisiones más informadas |
| Para el mercado | Mayor transparencia y eficiencia de precios |
| Para la academia | Dataset consolidado de alquileres en Lima (inédito) |
| Para política urbana | Identificación de zonas con sobreprecio estructural |

---

## 3. Disponibilidad, Calidad y Pertinencia de los Datos

### 3.1 Datos internos (a recolectar mediante scraping)

| Variable | Fuente | Estado |
|----------|--------|--------|
| Precio de alquiler mensual (S/.) | Urbania, Adondevivir | Por recolectar (scraping) |
| Área (m²) | Urbania, Adondevivir | Por recolectar |
| N.º de dormitorios y baños | Urbania, Adondevivir | Por recolectar |
| Piso del departamento | Urbania, Adondevivir | Por recolectar |
| Estacionamiento (sí/no) | Urbania, Adondevivir | Por recolectar |
| Amoblado (sí/no/parcial) | Urbania, Adondevivir | Por recolectar |
| Distrito y coordenadas GPS | Urbania, Adondevivir | Por recolectar |
| Antigüedad del edificio | Portales / inferencia | Por recolectar |

**Plan de recolección**: Se implementará un scraper en Python (BeautifulSoup + Selenium) que extraerá los listados activos de ambos portales. Se estima recolectar entre 8,000 y 15,000 registros de departamentos en Lima Metropolitana. Los datos serán almacenados en formato `.csv` y `.parquet`, y procesados con pandas.

### 3.2 Datos externos (fuentes públicas)

| Variable | Fuente | Estado |
|----------|--------|--------|
| Denuncias delictivas por distrito | PNP / INEI | Disponible (archivos públicos) |
| Establecimientos cercanos (supermercados, colegios, hospitales, parques) | Google Maps API / OpenStreetMap | Disponible (requiere API key) |
| Nivel socioeconómico por manzana | INEI Censos 2017 | Disponible |
| Líneas de transporte público | Protransporte / OSM | Disponible (parcial) |
| Calendario de eventos (ferias, eventos de alta demanda) | Fuentes varias | Secundario |

### 3.3 Datos nuevos a crear

A partir de las fuentes anteriores, el equipo construirá las siguientes variables derivadas:

- **Score de seguridad por zona** (0–100): índice compuesto basado en tasa de delitos por habitante, tipo de delito y proximidad a comisarías.
- **Índice de servicios cercanos**: cuenta ponderada de establecimientos útiles en un radio de 500 m y 1 km.
- **Precio estimado por m² por distrito**: estadística derivada del dataset scrapeado, útil como variable de referencia interna.
- **Indicador de precio justo**: variable binaria/continua que compara el precio publicado contra la predicción del modelo.
- **Ranking de zonas por relación precio-calidad**: puntuación compuesta que combina precio, seguridad y entorno.

### 3.4 Pertinencia y calidad esperada

Los datos de portales inmobiliarios presentan conocidas limitaciones: precios desactualizados, duplicados entre portales y campos incompletos. El plan de limpieza incluye: deduplicación por hash de descripción + coordenadas, imputación de valores faltantes (mediana por distrito), y validación de rangos (precio mínimo/máximo razonable por distrito).

---

## 4. Propuesta y Presentación del Producto

### Nombre del producto
**LLAVE EN MANO** — *Predice el precio justo, elige con inteligencia.*

### Abstract

LLAVE EN MANO es una plataforma de datos que predice el precio justo de alquiler de departamentos en Lima Metropolitana. Integra características físicas del inmueble, datos de seguridad por zona y equipamiento urbano cercano en un modelo predictivo (XGBoost/LightGBM). El producto permite al usuario ingresar los atributos de un departamento y obtener una estimación del precio de mercado, un score de calidad del entorno y una evaluación de si el precio publicado es justo o inflado.

### Background

El mercado de alquileres en Lima opera con marcada asimetría de información. Los portales inmobiliarios listan precios sin contexto comparativo, sin información de seguridad ni de servicios del entorno. No existe una herramienta pública que consolide estas tres dimensiones. Proyectos similares en mercados como Argentina (Properati) y España (Idealista Analytics) han demostrado el valor de los modelos predictivos de precios en el sector inmobiliario, pero ninguno se ha implementado para Lima.

### Propuesta general

El producto constará de tres módulos principales:

1. **Módulo de predicción de precio**: el usuario ingresa características del departamento y obtiene el precio estimado con intervalo de confianza.
2. **Módulo de evaluación de entorno**: visualización de score de seguridad y mapa de servicios cercanos para la dirección ingresada.
3. **Módulo de comparación de precio**: indica si el precio publicado en un portal es justo, sobrevaluado o subvaluado respecto a la predicción del modelo.

---

## 5. Usuarios Objetivo y Contexto de Uso

### Usuarios primarios

**Personas buscando alquilar un departamento en Lima**
- **Contexto**: buscan departamento con presupuesto definido, sin herramienta de referencia de precios.
- **Necesidad**: saber si el precio pedido es justo para la zona y las características del inmueble.
- **Cómo los afecta el problema**: pagan de más por desconocimiento; invierten mucho tiempo comparando manualmente.

### Usuarios secundarios

**Agentes inmobiliarios**
- **Contexto**: asesoran a clientes en la elección o publicación de departamentos.
- **Necesidad**: herramienta rápida para validar precios de mercado y fortalecer su propuesta de valor.
- **Cómo los afecta**: pérdida de credibilidad si el precio asesorado no es competitivo.

**Propietarios que desean arrendar**
- **Contexto**: fijan el precio de alquiler de su propiedad basándose en intuición o comparación manual.
- **Necesidad**: fijar un precio competitivo que no deje dinero sobre la mesa ni espante inquilinos.

### Stakeholders institucionales

**Investigadores urbanos / entidades de política de vivienda**
- Interés en datos consolidados de precios, seguridad y entorno para análisis de mercado y formulación de políticas.

### Sponsor

**Curso DS3022 – UTEC**: provee el marco académico, los recursos de infraestructura computacional y la validación del producto como entregable académico.

---

## 6. Data Product Canvas Inicial

| Sección | Contenido |
|---------|-----------|
| **1. Problema** | Falta de referencia objetiva para evaluar el precio de alquiler en Lima, considerando características del inmueble y su entorno. |
| **2. Usuario** | Personas buscando alquilar un departamento en Lima Metropolitana. |
| **3. Data** | Scraping de portales (Urbania, Adondevivir), datos PNP/INEI, Google Maps API, OpenStreetMap, censos INEI. |
| **4. Hipótesis** | Combinando características del inmueble + seguridad + entorno, el modelo puede predecir el precio con error < 10 % (MAPE). |
| **5. Solución** | Modelo predictivo (XGBoost) + score de entorno + plataforma web interactiva con comparador de precios. |
| **6. Actores** | Arrendatarios, agentes inmobiliarios, propietarios. Sponsor: UTEC / DPD. |
| **7. Métricas clave** | RMSE, MAE, MAPE, R². Score de seguridad por zona. NPS del producto. % de predicciones dentro del ±10 % del precio real. |
| **8. Impacto** | Mayor transparencia de precios. Reducción de sobrepago. Insumo para política urbana. Dataset público de alquileres en Lima. |
| **9. Acciones** | 1. Scraping de portales. 2. Recolección datos PNP/INEI/OSM. 3. EDA y construcción de variables. 4. Modelo base. 5. Prototipo web. 6. Validación con usuarios. |

---

## 7. Hipótesis Central

> **Si integramos características físicas del inmueble, datos de seguridad por zona (PNP/INEI) y disponibilidad de servicios cercanos (OSM/Google Maps), un modelo de machine learning puede predecir el precio de alquiler de departamentos en Lima con un error porcentual medio (MAPE) menor al 10 %, y generar un score de calidad de entorno reproducible y útil para la toma de decisiones.**

**Causa raíz del problema identificada**:
- Los arrendatarios no tienen acceso a información consolidada de precio + seguridad + entorno.
- No existe un modelo predictivo público para Lima.
- Los datos necesarios existen pero están dispersos en múltiples fuentes.

---

## 8. Tipo de Modelo

El producto utilizará tres tipos de análisis complementarios:

### Descriptivo
Exploración de patrones de precios por distrito, tipo de inmueble y características del entorno.
- **Técnicas**: estadística descriptiva, visualización geoespacial (Kepler.gl / Folium), análisis de correlación.
- **Outputs**: mapa de precios por zona, distribución de precios por características.

### Predictivo
Estimación del precio de alquiler a partir de las variables del inmueble y el entorno.
- **Técnica principal**: XGBoost / LightGBM (modelos de gradient boosting reconocidos por alto rendimiento en datos tabulares).
- **Baseline**: regresión lineal múltiple para comparar.
- **Validación**: cross-validation k-fold, holdout 80/20, métricas RMSE, MAE, MAPE, R².

### Prescriptivo
Generación de recomendaciones basadas en el modelo predictivo.
- **Score de precio justo**: indicador continuo que compara el precio publicado contra la predicción.
- **Ranking de zonas**: combinación de precio estimado, score de seguridad e índice de entorno para recomendar alternativas.

---

## 9. Métricas Clave

### Del modelo
- **RMSE** (Root Mean Square Error): error cuadrático medio del precio estimado en S/.
- **MAE** (Mean Absolute Error): error absoluto medio.
- **MAPE** (Mean Absolute Percentage Error): error porcentual medio — objetivo: < 10 %.
- **R²**: coeficiente de determinación — objetivo: > 0.85.

### Del producto
- Porcentaje de predicciones dentro del ±10 % del precio real observado.
- Precisión del score de seguridad validado contra datos PNP.
- Cobertura de distritos de Lima Metropolitana (objetivo: los 43 distritos).

### De impacto social
- Ahorro potencial estimado para el arrendatario en S/. por mes.
- Nivel de satisfacción del usuario (NPS — Net Promoter Score).
- Número de zonas con sobreprecio estructural detectadas.

---

## 10. Impacto Esperado

**Para el arrendatario**: mayor poder de negociación, reducción del sobrepago y decisiones basadas en evidencia. El producto democratiza información que antes solo tenían los agentes inmobiliarios.

**Para el mercado inmobiliario**: mayor eficiencia y transparencia de precios. Mejor planificación de zonas de inversión y alquiler basada en datos.

**Para la política urbana**: el dataset consolidado de alquileres, seguridad y entorno en Lima será un insumo inédito para la formulación de políticas de vivienda y planificación urbana.

