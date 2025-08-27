# 🏥 Medical Classification API

API REST simplificada para demostración de clasificación automática de literatura médica usando BioBERT y Gemini LLM en el challenge de programación.

## 🎯 Características Principales

- **🧠 Clasificación Híbrida**: Combina BioBERT fine-tuned con Gemini LLM para máxima precisión
- **📚 Dominios Médicos**: Cardiovascular, Neurológico, Oncológico, Hepatorenal
- **⚡ Procesamiento Rápido**: ~0.5s con BioBERT, ~2-5s con análisis LLM profundo
- **� Análisis Completo**: Métricas macro/micro/weighted, matrices de confusión
- **� Comparación de Modelos**: BioBERT vs LLM vs Sistema Híbrido
- **🚀 Demo Ready**: Sin autenticación, directo al grano para el concurso

## 🏗️ Arquitectura del Sistema

```
├── main.py                # API principal simplificada
├── .env                   # Configuración API keys
├── requirements.txt       # Dependencias
├── services/              # Lógica de clasificación
│   ├── biobert_classifier_enhanced.py    # BioBERT real
│   ├── llm_classifier_enhanced.py        # Gemini LLM real
│   ├── hybrid_classifier_enhanced.py     # Sistema híbrido
│   └── pipeline_enhanced.py              # Pipeline completo
├── utils/                 # Utilidades médicas
│   ├── medical_preprocessor.py           # Preprocesador
│   ├── medical_evaluator.py              # Evaluador de métricas
│   └── medical_label_analyzer.py         # Análisis de etiquetas
├── model/                 # Modelo BioBERT fine-tuned
│   └── biobert_finetuned_v3/
└── data/                  # Dataset del challenge
    └── raw/
        └── challenge_data-18-ago.csv
```

## 🚀 Inicio Rápido

### 1. Configuración del Entorno

```bash
# Navegar al proyecto
cd T1-datahack

# Instalar dependencias con uv
uv add fastapi uvicorn[standard] python-multipart pandas numpy torch transformers scikit-learn datasets python-dotenv google-generativeai

# O con pip
pip install -r requirements.txt
```

### 2. Configurar API Keys

```bash
# Ya tienes .env configurado con:
GEMINI_API_KEY=AIzaSyDNBFicMSLIWt50pkI2ux6sdFkx4kbzi0E
```

### 3. Ejecutar API

```bash
# Iniciar servidor
python main.py

# O con uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Verificar Instalación

```bash
# Health check
curl http://localhost:8000/

# Documentación interactiva
open http://localhost:8000/docs
```

## 🧪 Demo y Testing

```bash
# Test de salud de la API
curl http://localhost:8000/

# Subir dataset de prueba (usa el del challenge)
curl -X POST "http://localhost:8000/classify/upload-dataset" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/raw/challenge_data-18-ago.csv"

# Información de modelos cargados
curl http://localhost:8000/models/info
```

## 📚 Endpoints Principales

### 🏠 Estado de la API

- `GET /` - Health check y estado de modelos cargados

### 🧠 Clasificación

- `POST /classify/upload-dataset` - Subir CSV y clasificar con todos los modelos
- `GET /models/info` - Información detallada de los modelos cargados

### 📖 Documentación

- `GET /docs` - Documentación Swagger interactiva
- `GET /redoc` - Documentación ReDoc alternativa

## 🎯 Para el Concurso

### ✅ Funcionalidades Implementadas

- **Subir Dataset CSV** con estructura `title`, `abstract`, `group`
- **Análisis con 3 Modelos** simultáneamente:
  - BioBERT (modelo real fine-tuned)
  - LLM Gemini (limitado a 10 casos para demo)
  - Sistema Híbrido (combinación inteligente)
- **Métricas Completas**:
  - Accuracy, Precision, Recall, F1-Score
  - Macro, Micro y Weighted averaging
  - Matrices de confusión por etiqueta
- **Análisis de Distribución** de etiquetas
- **Estadísticas de Confianza** y rendimiento

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

### Subir Dataset y Obtener Análisis Completo

```python
import requests

# Subir CSV con estructura del challenge
with open('data/raw/challenge_data-18-ago.csv', 'rb') as f:
    response = requests.post(
        "http://localhost:8000/classify/upload-dataset",
        files={"file": f}
    )

result = response.json()

# El resultado incluye:
print("📊 Info del dataset:", result['dataset_info'])
print("🧬 Resultados BioBERT:", result['results']['biobert'])
print("🤖 Resultados LLM:", result['results']['llm'])
print("🔄 Resultados Híbrido:", result['results']['hybrid'])
print("📈 Análisis de etiquetas:", result['results']['label_analysis'])
```

### Ejemplo de Respuesta

```json
{
  "message": "Dataset procesado exitosamente",
  "dataset_info": {
    "total_articles": 50,
    "sample_titles": [
      "Effects of ACE inhibitors...",
      "Brain tumor classification..."
    ]
  },
  "results": {
    "biobert": {
      "model_name": "BioBERT Enhanced",
      "total_predictions": 50,
      "metrics": {
        "accuracy": 0.85,
        "precision_macro": 0.82,
        "recall_macro": 0.81,
        "f1_macro": 0.81
      },
      "confusion_matrices": {
        "cardiovascular": [
          [45, 5],
          [3, 47]
        ],
        "neurological": [
          [40, 10],
          [5, 45]
        ]
      }
    },
    "llm": {
      "model_name": "LLM (Gemini)",
      "total_predictions": 10,
      "metrics": {
        "accuracy": 0.9,
        "f1_macro": 0.88
      }
    },
    "hybrid": {
      "model_name": "Sistema Híbrido",
      "metrics": {
        "accuracy": 0.92,
        "f1_macro": 0.9
      }
    }
  }
}
```

## ⚙️ Configuración

### Estructura del Dataset CSV

El CSV debe tener exactamente estas columnas:

- `title`: Título del artículo médico
- `abstract`: Resumen científico del artículo
- `group`: Categoría(s) médica separadas por `|`

Ejemplo:

```csv
title;abstract;group
"Effects of ACE inhibitors";"This study examines...";"cardiovascular"
"Brain tumor analysis";"Machine learning approach...";"neurological|oncological"
```

## 🔧 Desarrollo

### Estructura de Proyecto Simplificada

```
T1-datahack/
├── main.py                              # API principal
├── .env                                 # API keys
├── requirements.txt                     # Dependencias
├── medical_classification_notebook.ipynb # Notebook original
├── services/                            # Lógica de modelos
│   ├── biobert_classifier_enhanced.py  # BioBERT real
│   ├── llm_classifier_enhanced.py      # Gemini LLM
│   ├── hybrid_classifier_enhanced.py   # Sistema híbrido
│   └── pipeline_enhanced.py            # Pipeline completo
├── utils/                               # Utilidades médicas
│   ├── medical_preprocessor.py          # Preprocesador
│   ├── medical_evaluator.py             # Evaluador
│   └── medical_label_analyzer.py        # Análisis etiquetas
├── model/                               # Modelo entrenado
│   └── biobert_finetuned_v3/
└── data/                                # Dataset challenge
    └── raw/
        └── challenge_data-18-ago.csv
```

### Dependencias Principales

- **FastAPI** - Framework web rápido
- **Transformers** - BioBERT y tokenización
- **PyTorch** - Deep learning
- **Google GenerativeAI** - Gemini LLM
- **Scikit-learn** - Métricas de evaluación
- **Pandas/NumPy** - Procesamiento de datos

## 📈 Performance y Métricas

### Tiempos de Procesamiento

- **BioBERT**: 0.3-0.7s por artículo
- **LLM Gemini**: 2-5s por artículo (limitado a 10 casos)
- **Sistema Híbrido**: 0.5-3s promedio
- **Dataset completo**: 30-50 artículos en ~20-30s

### Métricas Implementadas

#### Métricas de Clasificación

- **Accuracy** - Precisión exacta
- **Precision** (macro, micro, weighted)
- **Recall** (macro, micro, weighted)
- **F1-Score** (macro, micro, weighted)
- **Hamming Loss** - Para multilabel

#### Análisis Avanzado

- **Matrices de Confusión** por etiqueta
- **Distribución de Etiquetas** en el dataset
- **Estadísticas de Confianza** por modelo
- **Análisis de Casos Obvios vs Difíciles**

## 🏆 Funcionalidades del Challenge

### 🎯 Problema Multilabel

La API está diseñada específicamente para el problema de clasificación multilabel médica:

- **4 Dominios**: cardiovascular, hepatorenal, neurological, oncological
- **Múltiples etiquetas** por artículo permitidas
- **Evaluación robusta** con métricas apropiadas para multilabel

### 🧠 Enfoque Híbrido Inteligente

**BioBERT (90% casos obvios)**

- Rápido y eficiente
- Fine-tuned en tu dataset
- Maneja casos claros con alta confianza

**LLM Gemini (10% casos difíciles)**

- Análisis profundo con razonamiento
- Para casos ambiguos o complejos
- Limitado a 10 casos en la demo

**Sistema Híbrido**

- Routing automático basado en confianza
- Optimiza precisión vs velocidad
- Mejor rendimiento general

### 📊 Análisis Médico Especializado

En lugar de estadísticas básicas, la API proporciona:

- **Análisis de co-ocurrencia** entre dominios médicos
- **Distribución inteligente** de etiquetas
- **Métricas específicas** para literatura médica
- **Insights de complejidad** de casos

## 🐛 Troubleshooting

### Problemas Comunes

1. **Modelo BioBERT no encontrado**

   ```bash
   # Verifica que el modelo esté en la ruta correcta:
   # ./model/biobert_finetuned_v3/
   # Debe contener: config.json, model.safetensors, tokenizer.json, etc.
   ```

2. **Error con Gemini API**

   ```bash
   # Verifica que la API key esté en .env:
   GEMINI_API_KEY=tu_api_key_aqui

   # Si falla, el sistema usa fallback inteligente
   ```

3. **Error de dependencias**

   ```bash
   pip install --upgrade pip
   uv add fastapi uvicorn transformers torch scikit-learn
   ```

4. **Puerto ocupado**

   ```bash
   # Cambiar puerto en main.py línea final:
   uvicorn.run(app, host="0.0.0.0", port=8001)
   ```

5. **Dataset CSV mal formateado**
   ```bash
   # Verificar separador ; y columnas: title, abstract, group
   # Encoding UTF-8
   ```

### Logs y Debug

- **Consola**: Logs en tiempo real durante ejecución
- **Errores**: Se muestran claramente con traceback
- **Estado de modelos**: Verificar en GET /models/info

## 📞 Soporte y Uso

### 🚀 Para el Concurso

1. **Ejecutar API**: `python main.py`
2. **Abrir docs**: http://localhost:8000/docs
3. **Subir dataset**: POST /classify/upload-dataset
4. **Ver resultados**: JSON con métricas de los 3 modelos

### 🌐 Integración con v0

```javascript
// Para tu aplicación web en v0
const uploadDataset = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(
    "http://localhost:8000/classify/upload-dataset",
    {
      method: "POST",
      body: formData,
    }
  );

  return response.json();
};
```

### � Endpoints Clave

- **Health**: `GET /`
- **Upload**: `POST /classify/upload-dataset`
- **Models**: `GET /models/info`
- **Docs**: `GET /docs`

---

**🏥 Medical Classification API v2.0**  
_Clasificación inteligente de literatura médica para challenge de programación_
