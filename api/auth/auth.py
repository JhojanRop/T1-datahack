"""
Sistema de autenticación JWT para la API médica
"""
from datetime import datetime, timedelta
from typing import Any

import jwt
from fastapi import HTTPException, status
from passlib.context import CryptContext
from pydantic import BaseModel

# Configuración de encriptación
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Configuración JWT (en producción debe venir de variables de entorno)
SECRET_KEY = "medical-classification-api-secret-key-2025"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


class Token(BaseModel):
    """Modelo para token de respuesta"""
    access_token: str
    token_type: str
    expires_in: int


class TokenData(BaseModel):
    """Datos contenidos en el token"""
    username: str | None = None
    scopes: list[str] = []


class User(BaseModel):
    """Modelo de usuario"""
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool = False
    scopes: list[str] = []


class UserInDB(User):
    """Usuario en base de datos con hash de contraseña"""
    hashed_password: str


# Base de datos ficticia de usuarios (en producción usar DB real)
fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Administrator",
        "email": "admin@medical-api.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
        "disabled": False,
        "scopes": ["classification:read", "classification:write", "evaluation:read", "evaluation:write", "admin:read", "admin:write"]
    },
    "researcher": {
        "username": "researcher",
        "full_name": "Medical Researcher",
        "email": "researcher@medical-api.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
        "disabled": False,
        "scopes": ["classification:read", "classification:write", "evaluation:read"]
    },
    "viewer": {
        "username": "viewer",
        "full_name": "Data Viewer",
        "email": "viewer@medical-api.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
        "disabled": False,
        "scopes": ["classification:read", "evaluation:read"]
    }
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verificar contraseña"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Obtener hash de contraseña"""
    return pwd_context.hash(password)


def get_user(username: str) -> UserInDB | None:
    """Obtener usuario de la base de datos"""
    if username in fake_users_db:
        user_dict = fake_users_db[username]
        return UserInDB(**user_dict)
    return None


def authenticate_user(username: str, password: str) -> UserInDB | None:
    """Autenticar usuario"""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """Crear token de acceso JWT"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> TokenData:
    """Verificar y decodificar token JWT"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        scopes = payload.get("scopes", [])
        token_data = TokenData(username=username, scopes=scopes)
    except jwt.PyJWTError:
        raise credentials_exception

    return token_data


def get_current_user(token_data: TokenData) -> User:
    """Obtener usuario actual del token"""
    user = get_user(username=token_data.username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def get_current_active_user(current_user: User) -> User:
    """Verificar que el usuario esté activo"""
    if current_user.disabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


def check_user_permissions(user: User, required_scope: str) -> bool:
    """Verificar permisos del usuario"""
    return required_scope in user.scopes


def require_scope(required_scope: str):
    """Decorador para requerir un scope específico"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # En una implementación real, extraería el usuario del contexto
            # Por ahora es un placeholder
            return func(*args, **kwargs)
        return wrapper
    return decorator
