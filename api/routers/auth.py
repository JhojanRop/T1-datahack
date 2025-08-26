"""
Router de autenticación - Login y gestión de tokens
"""
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from api.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    authenticate_user,
    create_access_token,
)
from core import log_security_event, logger

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])


class LoginResponse(BaseModel):
    """Respuesta de login exitoso"""
    access_token: str
    token_type: str
    expires_in: int
    user_info: dict[str, str]


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Autenticación de usuario",
    description="Autentica usuario y retorna token JWT para acceso a la API"
)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Endpoint de login"""
    try:
        # Autenticar usuario
        user = authenticate_user(form_data.username, form_data.password)
        if not user:
            log_security_event(
                "failed_login",
                {"username": form_data.username, "ip": "unknown"},
                "WARNING"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Usuario o contraseña incorrectos",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Crear token de acceso
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username, "scopes": user.scopes},
            expires_delta=access_token_expires
        )

        # Log de login exitoso
        log_security_event(
            "successful_login",
            {"username": user.username, "scopes": user.scopes},
            "INFO"
        )

        logger.info(f"Usuario autenticado: {user.username}")

        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user_info={
                "username": user.username,
                "full_name": user.full_name or "",
                "email": user.email or "",
                "scopes": user.scopes
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor"
        ) from e


@router.get(
    "/users/demo",
    summary="Usuarios de demostración",
    description="Lista los usuarios disponibles para testing (solo para desarrollo)"
)
async def get_demo_users():
    """Obtener usuarios de demostración disponibles"""
    return {
        "demo_users": [
            {
                "username": "admin",
                "password": "secret",
                "scopes": ["classification:read", "classification:write", "evaluation:read", "evaluation:write", "admin:read", "admin:write"],
                "description": "Usuario administrador con acceso completo"
            },
            {
                "username": "researcher",
                "password": "secret",
                "scopes": ["classification:read", "classification:write", "evaluation:read"],
                "description": "Investigador médico con acceso a clasificación y evaluación"
            },
            {
                "username": "viewer",
                "password": "secret",
                "scopes": ["classification:read", "evaluation:read"],
                "description": "Usuario con acceso de solo lectura"
            }
        ],
        "note": "Estos usuarios son solo para demostración. En producción, usar usuarios reales con contraseñas seguras."
    }
