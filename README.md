# 🏥 Medical Literature Classification System

Un sistema avanzado de **clasificación automática de literatura médica** usando técnicas de Deep Learning y modelos de lenguaje de gran escala (LLM).

## 🎯 Descripción del Proyecto

Este proyecto implementa una solución completa de **Inteligencia Artificial** para la clasificación automática de artículos médicos en dominios especializados. Combina la potencia de **BioBERT** fine-tuned con **Gemini LLM** para proporcionar clasificaciones precisas y confiables.

### 🧠 Dominios de Clasificación

- **💓 Cardiovascular**: Cardiología, cirugía cardíaca, enfermedades vasculares
- **🧠 Neurológico**: Neurología, neurocirugía, enfermedades neurodegenerativas
- **🎗️ Oncológico**: Oncología, cáncer, tratamientos oncológicos
- **🫘 Hepatorenal**: Hepatología, nefrología, trasplantes

## 🚀 Características Principales

- ✅ **API REST FastAPI** con documentación automática
- ✅ **Clasificación Híbrida** BioBERT + Gemini LLM
- ✅ **Autenticación JWT** con roles y permisos
- ✅ **Procesamiento en lote** optimizado
- ✅ **Jupyter Notebook** interactivo incluido
- ✅ **Sistema de confianza** robusto con múltiples métricas
- ✅ **Scripts de demostración** y testing
- ✅ **Logs avanzados** con rotación automática

## 📊 Performance

| Método         | Tiempo Promedio | Casos de Uso                        |
| -------------- | --------------- | ----------------------------------- |
| **BioBERT**    | ~0.5s           | Clasificación rápida general        |
| **Gemini LLM** | ~2-5s           | Análisis profundo y casos complejos |
| **Híbrido**    | ~0.5-3s         | Routing inteligente automático      |

## 🏗️ Arquitectura del Sistema

```
T1-datahack/
├── 📊 medical_classification_notebook.ipynb  # Análisis y experimentación
├── 🚀 api_main.py                           # API principal
├── 🧪 demo_api.py                           # Script de demostración
├── ⚡ start_api.py                          # Setup automático
├── api/                                     # Módulos de la API
│   ├── auth/                               # Sistema de autenticación
│   ├── models/                             # Modelos Pydantic
│   └── routers/                            # Endpoints REST
├── services/                               # Clasificadores IA
│   ├── biobert_classifier_enhanced.py      # Clasificador BioBERT
│   ├── llm_classifier_enhanced.py          # Clasificador LLM
│   └── hybrid_classifier_enhanced.py       # Sistema híbrido
├── core/                                   # Configuración
│   ├── config.py                          # Settings del sistema
│   └── logging.py                         # Sistema de logs
├── model/                                  # Modelos entrenados
│   └── biobert_finetuned_v3/              # BioBERT fine-tuned
└── data/                                   # Datasets y cache
    └── raw/challenge_data-18-ago.csv       # Dataset principal
```

## 🚀 Inicio Rápido

### 1. **Setup Automático** (Recomendado)

```bash
# Clonar el repositorio
git clone https://github.com/JhojanRop/T1-datahack.git
cd T1-datahack

# Ejecutar setup automático
python start_api.py
```

### 2. **Setup Manual**

```bash
# Instalar dependencias
pip install -e .
# O con UV:
uv sync

# Configurar entorno
cp .env.example .env
# Editar .env con tus configuraciones

# Crear directorios necesarios
mkdir -p logs data/temp data/processed data/cache

# Iniciar API
python api_main.py
```

### 3. **Verificar Instalación**

```bash
# Health check
curl http://localhost:8000/health

# Documentación interactiva
open http://localhost:8000/docs
```

## 📚 Uso del Sistema

### 🌐 **API REST**

#### Autenticación

```bash
# Login (usuarios demo disponibles)
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=secret"
```

#### Clasificación Individual

```bash
# Clasificar un artículo
curl -X POST "http://localhost:8000/api/v1/classification/single" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Effects of ACE inhibitors on cardiovascular outcomes",
    "abstract": "This study examines the cardiovascular benefits...",
    "authors": "Smith J., Johnson A.",
    "journal": "Journal of Cardiology"
  }'
```

#### Clasificación en Lote

```bash
# Procesar múltiples artículos
curl -X POST "http://localhost:8000/api/v1/classification/batch" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "articles": [...],
    "method": "hybrid",
    "parallel_processing": true
  }'
```

### 📓 **Jupyter Notebook**

```bash
# Iniciar Jupyter
jupyter lab medical_classification_notebook.ipynb
```

El notebook incluye:

- 📊 Análisis exploratorio de datos
- 🧠 Entrenamiento de modelos
- 📈 Evaluación de performance
- 🔬 Experimentación interactiva

### 🧪 **Demo y Testing**

```bash
# Ejecutar demostración completa
python demo_api.py

# El script incluye:
# - Test de autenticación
# - Clasificación individual
# - Procesamiento en lote
# - Análisis de performance
```

## 👤 Usuarios Demo

| Usuario      | Contraseña | Permisos                      |
| ------------ | ---------- | ----------------------------- |
| `admin`      | `secret`   | 🔐 Acceso completo            |
| `researcher` | `secret`   | 🔬 Clasificación + Evaluación |
| `viewer`     | `secret`   | 👁️ Solo lectura               |

## 🔧 Configuración

### Variables de Entorno Principales

```bash
# Modelo y IA
BIOBERT_MODEL_PATH=./model/biobert_finetuned_v3
GEMINI_API_KEY=your_gemini_api_key_here
CONFIDENCE_THRESHOLD=0.7

# Servidor
PORT=8000
DEBUG=true
ENVIRONMENT=development

# Seguridad
SECRET_KEY=your_secret_key_here
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

## 📈 Endpoints Principales

### 🔐 Autenticación

- `POST /api/v1/auth/login` - Login con JWT
- `GET /api/v1/auth/users/demo` - Usuarios de demostración

### 🧠 Clasificación

- `POST /api/v1/classification/single` - Clasificar artículo individual
- `POST /api/v1/classification/batch` - Clasificación en lote
- `POST /api/v1/classification/quick-test` - Test rápido
- `GET /api/v1/classification/methods` - Métodos disponibles
- `GET /api/v1/classification/domains` - Dominios médicos

### ℹ️ Sistema

- `GET /health` - Health check del sistema
- `GET /api/v1/info` - Información detallada de la API
- `GET /docs` - Documentación Swagger
- `GET /redoc` - Documentación ReDoc

## 🧠 Modelos de IA

### 1. **BioBERT Fine-tuned**

- **Base**: `dmis-lab/biobert-base-cased-v1.1`
- **Especialización**: Literatura médica en 4 dominios
- **Performance**: ~0.5s por clasificación
- **Uso**: Clasificación rápida y eficiente

### 2. **Gemini LLM**

- **Modelo**: `gemini-2.0-flash`
- **Capacidades**: Razonamiento profundo y análisis contextual
- **Performance**: ~2-5s por clasificación
- **Uso**: Casos complejos y análisis detallado

### 3. **Sistema Híbrido**

- **Routing Inteligente**: Selección automática del mejor modelo
- **Criterios**: Confianza, complejidad, recursos disponibles
- **Optimización**: Balance óptimo entre precisión y velocidad

## 📊 Métricas de Confianza

El sistema utiliza múltiples métricas para evaluar la confianza:

- **Softmax Probabilities**: Distribución de probabilidades
- **Entropía Normalizada**: Medida de incertidumbre
- **Top-2 Difference**: Diferencia entre las dos predicciones principales
- **Keywords Medical**: Presencia de términos médicos específicos
- **Combinación Ponderada**: Score final robusto

## 🔍 Análisis y Monitoreo

### Logging Avanzado

- 📝 **Logs estructurados** con niveles de severidad
- 🌈 **Colores por tipo** de evento
- 🔄 **Rotación automática** de archivos
- 📊 **Métricas de performance** integradas

### Health Monitoring

- ⚡ **Health checks** automáticos
- 📈 **Métricas de sistema** en tiempo real
- 🚨 **Alertas** por errores críticos
- 📱 **Status dashboard** en `/health`

## 🛠️ Desarrollo

### Dependencias Principales

```toml
[dependencies]
fastapi = "^0.104.1"
transformers = "^4.35.2"
torch = "^2.1.0"
google-generativeai = "^0.3.0"
uvicorn = "^0.24.0"
pydantic = "^2.5.0"
python-jose = "^3.3.0"
python-multipart = "^0.0.6"
pandas = "^2.1.3"
numpy = "^1.24.3"
scikit-learn = "^1.3.2"
```

### Scripts de Utilidad

- `start_api.py` - Setup automático y arranque del servidor
- `demo_api.py` - Demostración completa con todos los casos de uso
- `api_main.py` - Aplicación principal FastAPI
- `main.py` - Script principal alternativo

## 📈 Performance y Escalabilidad

### Optimizaciones Implementadas

- ✅ **Caching inteligente** de predicciones
- ✅ **Procesamiento paralelo** para lotes
- ✅ **Rate limiting** por usuario
- ✅ **Background tasks** para operaciones pesadas
- ✅ **Memory optimization** en modelos

### Métricas de Rendimiento

- **Throughput**: ~100-200 clasificaciones/minuto
- **Latencia**: 0.5-3s dependiendo del método
- **Memory Usage**: ~2-4GB con modelos cargados
- **CPU Usage**: Optimizado para multi-core

## 🐛 Troubleshooting

### Problemas Comunes

1. **Modelo BioBERT no encontrado**

   ```bash
   # El sistema funcionará en modo simulación
   # Descargar y colocar en: ./model/biobert_finetuned_v3/
   ```

2. **Error de API Key de Gemini**

   ```bash
   # Obtener key en: https://makersuite.google.com/app/apikey
   # Configurar en .env: GEMINI_API_KEY=tu_key_aqui
   ```

3. **Puerto ocupado**
   ```bash
   # Cambiar puerto en .env o usar:
   uvicorn api_main:app --port 8001
   ```

## 📖 Documentación Adicional

- 📚 **[README_API.md](README_API.md)** - Documentación detallada de la API
- 📓 **[Jupyter Notebook](medical_classification_notebook.ipynb)** - Análisis interactivo
- 🌐 **[Swagger UI](http://localhost:8000/docs)** - Documentación interactiva
- 📑 **[ReDoc](http://localhost:8000/redoc)** - Documentación alternativa

## 🤝 Contribución

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crea un Pull Request

## 📄 Licencia

Este proyecto está bajo la licencia especificada en el archivo [LICENSE](LICENSE).

## 🏆 Reconocimientos

- **BioBERT**: [dmis-lab/biobert](https://github.com/dmis-lab/biobert)
- **Google Gemini**: [Google AI](https://ai.google.dev/)
- **FastAPI**: [FastAPI Framework](https://fastapi.tiangolo.com/)
- **Hugging Face**: [Transformers Library](https://huggingface.co/transformers/)

---

**🏥 Medical Literature Classification System v1.0.0**  
_Desarrollado para la clasificación inteligente de literatura médica usando IA avanzada_

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![BioBERT](https://img.shields.io/badge/BioBERT-Fine--tuned-orange.svg)](https://github.com/dmis-lab/biobert)
[![Gemini](https://img.shields.io/badge/Gemini-2.0--flash-red.svg)](https://ai.google.dev)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
