"""
Middleware de autenticación para FastAPI
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .auth import (
    TokenData,
    User,
    check_user_permissions,
    get_current_active_user,
    get_current_user,
    verify_token,
)

# Esquema de seguridad Bearer
security = HTTPBearer()


async def get_current_user_dependency(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Dependency para obtener el usuario actual"""
    token = credentials.credentials
    token_data = verify_token(token)
    user = get_current_user(token_data)
    return get_current_active_user(user)


def require_permission(required_scope: str):
    """Dependency factory para requerir permisos específicos"""
    async def permission_checker(current_user: User = Depends(get_current_user_dependency)):
        if not check_user_permissions(current_user, required_scope):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {required_scope}"
            )
        return current_user
    return permission_checker


# Dependencies predefinidas para diferentes niveles de acceso
require_classification_read = require_permission("classification:read")
require_classification_write = require_permission("classification:write")
require_evaluation_read = require_permission("evaluation:read")
require_evaluation_write = require_permission("evaluation:write")
require_admin_read = require_permission("admin:read")
require_admin_write = require_permission("admin:write")


class AuthMiddleware:
    """Middleware personalizado para autenticación"""

    def __init__(self):
        self.public_routes = {
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/auth/login",
            "/api/v1/health"
        }

    async def __call__(self, request, call_next):
        """Procesar request de autenticación"""
        # Permitir rutas públicas
        if request.url.path in self.public_routes:
            return await call_next(request)

        # Verificar header de autorización
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Extraer y verificar token
        token = auth_header.split(" ")[1]
        try:
            token_data = verify_token(token)
            user = get_current_user(token_data)
            get_current_active_user(user)

            # Agregar usuario al estado del request
            request.state.current_user = user

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token validation failed",
                headers={"WWW-Authenticate": "Bearer"},
            ) from e

        return await call_next(request)
