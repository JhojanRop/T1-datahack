# 🏥 Medical Classification API

Una API REST avanzada para clasificación automática de literatura médica usando BioBERT y modelos LLM.

## 🎯 Características Principales

- **🧠 Clasificación Híbrida**: Combina BioBERT fine-tuned con Gemini LLM para máxima precisión
- **📚 Dominios Médicos**: Cardiovascular, Neurológico, Oncológico, Hepatorenal
- **⚡ Procesamiento Rápido**: ~0.5s con BioBERT, ~2-5s con análisis LLM profundo
- **📦 Batch Processing**: Procesamiento en lote con paralelización
- **🔐 Autenticación JWT**: Sistema de permisos granular con scopes
- **📊 Métricas Avanzadas**: Scores de confianza robustos y metadatos detallados
- **🚀 FastAPI**: Documentación automática y alta performance

## 🏗️ Arquitectura del Sistema

```
├── api/                    # API REST endpoints
│   ├── auth/              # Autenticación JWT
│   ├── models/            # Modelos Pydantic
│   └── routers/           # Rutas organizadas
├── services/              # Lógica de clasificación
│   ├── biobert_classifier_enhanced.py
│   ├── llm_classifier_enhanced.py
│   └── hybrid_classifier_enhanced.py
├── core/                  # Configuración y logging
├── model/                 # Modelo BioBERT fine-tuned
└── data/                  # Datos y cache
```

## 🚀 Inicio Rápido

### 1. Configuración del Entorno

```bash
# Clonar y navegar al proyecto
cd T1-datahack

# Ejecutar script de configuración automática
python start_api.py
```

### 2. Configuración Manual (Alternativa)

```bash
# Copiar configuración
cp .env.example .env

# Instalar dependencias
pip install -e .
# O con UV:
uv sync

# Crear directorios
mkdir -p logs data/temp data/processed data/cache

# Iniciar API
python api_main.py
# O:
uvicorn api_main:app --reload
```

### 3. Verificar Instalación

```bash
# Health check
curl http://localhost:8000/health

# Documentación interactiva
open http://localhost:8000/docs
```

## 🧪 Demo y Testing

```bash
# Ejecutar demostración completa
python demo_api.py

# Test manual con curl
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=secret"
```

## 📚 Endpoints Principales

### 🔐 Autenticación

- `POST /api/v1/auth/login` - Login con JWT
- `GET /api/v1/auth/users/demo` - Usuarios demo

### 🧠 Clasificación

- `POST /api/v1/classification/single` - Clasificar artículo individual
- `POST /api/v1/classification/batch` - Clasificación en lote
- `POST /api/v1/classification/quick-test` - Test rápido
- `GET /api/v1/classification/methods` - Métodos disponibles
- `GET /api/v1/classification/domains` - Dominios médicos

### ℹ️ Información

- `GET /health` - Health check
- `GET /api/v1/info` - Información detallada de la API
- `GET /docs` - Documentación Swagger
- `GET /redoc` - Documentación ReDoc

## 👤 Usuarios Demo

| Usuario      | Contraseña | Permisos                   |
| ------------ | ---------- | -------------------------- |
| `admin`      | `secret`   | Acceso completo            |
| `researcher` | `secret`   | Clasificación + Evaluación |
| `viewer`     | `secret`   | Solo lectura               |

## 🧠 Métodos de Clasificación

### 1. BioBERT (Rápido)

- **Tiempo**: ~0.5s
- **Uso**: Clasificación general rápida
- **Confianza**: Análisis multi-método

### 2. LLM/Gemini (Profundo)

- **Tiempo**: ~2-5s
- **Uso**: Casos complejos o ambiguos
- **Características**: Razonamiento explícito

### 3. Híbrido (Recomendado)

- **Tiempo**: ~0.5-3s
- **Uso**: Routing inteligente automático
- **Optimización**: Mejor balance precisión/velocidad

## 📊 Ejemplo de Uso

### Clasificación Individual

```python
import requests

# Login
login_response = requests.post(
    "http://localhost:8000/api/v1/auth/login",
    data={"username": "admin", "password": "secret"}
)
token = login_response.json()["access_token"]

# Clasificar artículo
headers = {"Authorization": f"Bearer {token}"}
article = {
    "title": "Effects of ACE inhibitors on cardiovascular outcomes",
    "abstract": "This study examines the cardiovascular benefits...",
    "authors": "Smith J., Johnson A.",
    "journal": "Journal of Cardiology"
}

response = requests.post(
    "http://localhost:8000/api/v1/classification/single",
    json=article,
    headers=headers,
    params={"method": "hybrid"}
)

result = response.json()
print(f"Dominios: {result['domains']}")
print(f"Confianza: {result['confidence_scores']}")
print(f"Método: {result['method_used']}")
```

### Clasificación en Lote

```python
batch_request = {
    "articles": [
        {
            "title": "Cardiac surgery outcomes",
            "abstract": "Analysis of cardiac surgery..."
        },
        {
            "title": "Brain tumor classification",
            "abstract": "ML approaches for brain tumor..."
        }
    ],
    "method": "hybrid",
    "confidence_threshold": 0.7,
    "parallel_processing": True
}

response = requests.post(
    "http://localhost:8000/api/v1/classification/batch",
    json=batch_request,
    headers=headers
)
```

## ⚙️ Configuración

### Variables de Entorno (.env)

```bash
# Servidor
ENVIRONMENT=development
HOST=0.0.0.0
PORT=8000
DEBUG=true

# Modelo
BIOBERT_MODEL_PATH=./model/biobert_finetuned_v3
CONFIDENCE_THRESHOLD=0.7
LLM_THRESHOLD=0.6

# Gemini LLM (opcional)
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.0-flash

# Seguridad
SECRET_KEY=your_secret_key_here
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Pipeline
MAX_BATCH_SIZE=50
PARALLEL_PROCESSING=true
ENABLE_CACHING=true
```

### Obtener API Key de Gemini

1. Visita [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Crea una nueva API key
3. Agrega la key al archivo `.env`

## 🔧 Desarrollo

### Estructura de Proyecto

```
T1-datahack/
├── api/                    # REST API
│   ├── auth/              # Autenticación
│   ├── models/            # Modelos Pydantic
│   └── routers/           # Endpoints
├── services/              # Clasificadores
├── core/                  # Config + Logging
├── model/                 # BioBERT modelo
├── data/                  # Datasets
├── utils/                 # Utilidades
├── api_main.py           # App principal
├── demo_api.py           # Script demo
├── start_api.py          # Setup automático
└── pyproject.toml        # Dependencias
```

### Dependencias Principales

- **FastAPI** - Framework web
- **Transformers** - BioBERT
- **PyTorch** - Deep learning
- **Google GenerativeAI** - Gemini LLM
- **Pydantic** - Validación datos
- **Uvicorn** - Servidor ASGI

## 📈 Performance y Optimización

### Métricas Típicas

- **BioBERT**: 0.3-0.7s por artículo
- **LLM**: 2-5s por artículo
- **Híbrido**: 0.5-3s (promedio)
- **Batch**: Escalabilidad lineal

### Optimizaciones Implementadas

- ✅ Caching inteligente
- ✅ Procesamiento paralelo
- ✅ Rate limiting
- ✅ Routing automático
- ✅ Background tasks
- ✅ Memory optimization

## 🏆 Características Avanzadas

### Sistema Híbrido Inteligente

El clasificador híbrido decide automáticamente cuándo usar LLM basado en:

- **Confianza baja** de BioBERT (< 0.6)
- **Alta entropía** en predicciones
- **Múltiples dominios** con confianza similar
- **Texto inusual** (muy corto/largo)

### Análisis de Confianza Robusto

- **Softmax probabilities**
- **Entropía normalizada**
- **Diferencia top-2**
- **Keywords médicas**
- **Combinación ponderada**

### Logging y Monitoreo

- **Logs estructurados** con colores
- **Rotación automática** de archivos
- **Métricas de performance**
- **Eventos de seguridad**
- **Request tracking**

## 🐛 Troubleshooting

### Problemas Comunes

1. **Modelo BioBERT no encontrado**

   ```bash
   # El sistema funcionará en modo simulación
   # Coloca el modelo en: ./model/biobert_finetuned_v3/
   ```

2. **Error de dependencias**

   ```bash
   pip install --upgrade pip
   pip install -e .
   ```

3. **Puerto ocupado**

   ```bash
   # Cambiar puerto en .env
   PORT=8001
   ```

4. **Error de permisos**
   ```bash
   # Verificar token JWT válido
   # Verificar scopes del usuario
   ```

## 📞 Soporte

- **Documentación**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Demo Script**: `python demo_api.py`
- **Logs**: `./logs/medical_api.log`

## 📄 Licencia

Este proyecto está bajo la licencia especificada en el archivo LICENSE.

---

**🏥 Medical Classification API v1.0.0**  
_Clasificación inteligente de literatura médica con IA_
