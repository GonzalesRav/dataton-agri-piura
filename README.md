# 🌾 AgriPiura — Dashboard de datos abiertos para la planificación agrícola

Proyecto desarrollado para la **Datatón 2025 - Perú**, usando datasets abiertos del **Gobierno Regional de Piura**.  
El objetivo es brindar una herramienta sencilla, accesible y transparente que apoye a agricultores, extensionistas y autoridades en la **planificación de campañas agrícolas**.

---

## 🚀 Descripción
**AgriPiura** es un **dashboard interactivo** construido con **software libre** (Python, Streamlit, Plotly, Pandas).  
Permite explorar y visualizar datos agrícolas de forma amigable, con indicadores clave como:

- 🗺️ **Mapa distrital** con índices de **flexibilidad** y **diversificación**.  
- 📊 **Ranking de cultivos transitorios** por superficie sembrada.  
- 📈 **Tendencias históricas** de área y precio por cultivo.  
- 🍩 **Dona de flexibilidad** (transitorios vs. permanentes).  

Cada gráfico incluye **notas metodológicas** que explican cómo interpretarlo y qué limitaciones considerar, para garantizar la transparencia y un uso responsable de los datos.

---

## ⚙️ Tecnologías utilizadas
- [Python 3.x](https://www.python.org/)  
- [Streamlit](https://streamlit.io/)  
- [Plotly](https://plotly.com/python/)  
- [Pandas](https://pandas.pydata.org/)  
- [GeoJSON](https://geojson.org/)

---

## 📂 Estructura del proyecto
├── app.py # Código principal del dashboard
├── requirements.txt # Dependencias del proyecto
├── data_proc/ # Datos preprocesados
│ ├── dataset_piura_clean.csv
│ └── dataset_piura_anual.csv
│ └── preprocess_report.md
├── geo/ # Capas geográficas
│ └── distritos_piura.geojson
├── scripts/ # Scripts de preprocesamiento
│ └── preprocess_piura.py
└── README.md
└── LICENSE

---

## 🔎 Cómo reproducirlo

1. **Clonar el repositorio**  
   ```bash
   git clone https://github.com/GonzalesRav/dataton-agri-piura
   cd dataton-agri-piura
   ```

2. **Instalar dependencias**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecutar el dashboard**  
   ```bash
   streamlit run app.py
   ```

---

## 🌍 Objetivos de Desarrollo Sostenible

El proyecto contribuye a:

- **ODS 2: Hambre cero**, mejorando la planificación de campañas agrícolas.  
- **ODS 12: Producción y consumo responsables**, reduciendo riesgos de sobreoferta.  
- **ODS 8: Trabajo decente y crecimiento económico**, apoyando ingresos más estables para agricultores.  

---

## 👩‍💻 Autora

**Johana Gonzales Ravenna**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/gonzalesrav/)  

---

## 📜 Licencia

Este proyecto se distribuye bajo licencia **MIT**.  
El código es **abierto y reproducible** para fomentar la innovación en la agricultura sostenible.  

---

## 🛠️ Badges

![Made with Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-FF4B4B?logo=streamlit&logoColor=white)  
![Open Data](https://img.shields.io/badge/Open%20Data-%F0%9F%93%84-green)  
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)  