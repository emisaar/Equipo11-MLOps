## 1) Estructuración de Proyectos con Cookiecutter
### Tarea: Implementar una estructura de proyecto estandarizada.

### Instrucciones:

- Descarga y utiliza la plantilla de CookiecutterLinks to an external site. para proyectos de ML.

- Implementa el esquema de directorios y archivos propuesto en tu propio proyecto siguiendo la plantilla de Cookiecutter

- Asegúrate de mantener una organización clara y consistente, de acuerdo al template.

- Importancia: Una buena estructura de proyecto facilita la colaboración, el mantenimiento y la escalabilidad de los desarrollos en ML.

### Estructura de CookieCutter:
```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         {{ cookiecutter.module_name }} and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── {{ cookiecutter.module_name }}   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes {{ cookiecutter.module_name }} a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```


## 2) Estructuración y Refactorización del Código
### Tarea: Mejorar la organización y mantenibilidad del código.

### Instrucciones:

- Organiza el código en módulos y funciones con responsabilidades bien definidas.

- Aplica principios de programación orientada a objetos (POO) cuando sea pertinente.

- Refactoriza el código existente para mejorar su eficiencia, legibilidad, escalabiliad y mantenimiento a largo plazo.

- Importancia: Un código bien estructurado y modularizado es clave para proyectos de ML que evolucionan en el tiempo.

 
## 3) Aplicación de Mejores Prácticas de Codificación en el Pipeline de Modelado
### Tarea: Incorporar buenas prácticas en las etapas del pipeline usando SciKit-Learn.

### Instrucciones:

- Implementa un pipeline de Scikit-Learn que automatice las etapas de preprocesamiento, entrenamiento y evaluación.

- Documenta cada paso, asegurando que sea claro, reproducible y entendible por terceros.

- Importancia: Seguir mejores prácticas garantiza que los proyectos sean eficientes, confiables y fáciles de replicar.

 

## 4) Seguimiento de Experimentos, Visualización de Resultados y Gestión de Modelos
### Tarea: Registrar y versionar experimentos, visualizar resultados y gestionar modelos de forma ordenada.

### Instrucciones:

- Utiliza herramientas como MLflow, DVC para dar seguimiento a los experimentos.

- Documenta y compara configuraciones, parámetros y resultados de cada ejecución.

- Registra métricas relevantes y asegura el control de versiones de los experimentos.

- Utiliza las herramientas de visualización de MLFlow  para presentar los resultados de manera clara y comprensible.

- Mantén un registro actualizado de los modelos generados, incluyendo:

    - Versión

    - Hiperparámetros

    - Métricas de evaluación

    - Resultados relevantes

- Importancia: Un registro sistemático de los experimentos y modelos, acompañado de una visualización clara (pueden usar las gráficas de comparativas de experimentos de MLFlow), que permite  el análisis comparativo, la toma de decisiones informadas y una gestión profesional de los proyectos.