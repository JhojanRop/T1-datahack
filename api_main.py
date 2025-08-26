"""
Aplicación principal FastAPI para la API de clasificación médica
"""
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routers import auth, classification
from core import logger, settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida de la aplicación"""
    # Startup
    logger.info("🚀 Iniciando API de Clasificación Médica")
    logger.info(f"📊 Configuración: {settings.app_name} v{settings.app_version}")
    logger.info(f"🌐 Entorno: {settings.__class__.__name__}")

    yield

    # Shutdown
    logger.info("🛑 Cerrando API de Clasificación Médica")


app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
)


# Middleware de logging de requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware para logging de requests"""
    start_time = time.time()

    # Log request
    logger.info(f"📥 {request.method} {request.url.path} - {request.client.host if request.client else 'unknown'}")

    # Procesar request
    response = await call_next(request)

    # Log response
    process_time = time.time() - start_time
    logger.info(f"📤 {request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")

    # Agregar headers de tiempo
    response.headers["X-Process-Time"] = str(process_time)

    return response


# Middleware de manejo de errores
@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    """Manejo de errores internos"""
    logger.error(f"❌ Error interno: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "Ha ocurrido un error interno en el servidor",
            "timestamp": time.time()
        }
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception):
    """Manejo de rutas no encontradas"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"Ruta no encontrada: {request.url.path}",
            "timestamp": time.time()
        }
    )


# Incluir routers
app.include_router(auth.router)
app.include_router(classification.router)


# Endpoints base
@app.get("/", tags=["root"])
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "🏥 Medical Classification API",
        "version": settings.app_version,
        "description": settings.app_description,
        "status": "active",
        "docs": "/docs",
        "endpoints": {
            "authentication": "/api/v1/auth/login",
            "classification": "/api/v1/classification/single",
            "batch_classification": "/api/v1/classification/batch",
            "quick_test": "/api/v1/classification/quick-test"
        },
        "timestamp": time.time()
    }


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "medical-classification-api",
        "version": settings.app_version,
        "timestamp": time.time(),
        "uptime": "active"
    }


@app.get("/api/v1/info", tags=["info"])
async def api_info():
    """Información detallada de la API"""
    return {
        "api_name": settings.app_name,
        "version": settings.app_version,
        "description": settings.app_description,
        "capabilities": {
            "biobert_classification": True,
            "llm_classification": True,
            "hybrid_classification": True,
            "batch_processing": True,
            "multilabel_classification": True
        },
        "medical_domains": [
            "cardiovascular",
            "neurological",
            "oncological",
            "hepatorenal"
        ],
        "authentication": {
            "type": "JWT Bearer Token",
            "scopes": [
                "classification:read",
                "classification:write",
                "evaluation:read",
                "evaluation:write",
                "admin:read",
                "admin:write"
            ]
        },
        "rate_limits": {
            "requests_per_minute": settings.rate_limit_requests,
            "window_seconds": settings.rate_limit_window
        },
        "contact": {
            "documentation": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        },
        "timestamp": time.time()
    }


if __name__ == "__main__":
    # Ejecutar servidor de desarrollo
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
