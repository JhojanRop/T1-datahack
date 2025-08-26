"""
Módulo de autenticación para la API médica
"""
from .auth import (
    Token,
    TokenData,
    User,
    UserInDB,
    authenticate_user,
    check_user_permissions,
    create_access_token,
    get_current_active_user,
    get_current_user,
    verify_token,
)
from .middleware import (
    AuthMiddleware,
    get_current_user_dependency,
    require_admin_read,
    require_admin_write,
    require_classification_read,
    require_classification_write,
    require_evaluation_read,
    require_evaluation_write,
    require_permission,
)

__all__ = [
    # Auth functions
    "authenticate_user",
    "create_access_token",
    "verify_token",
    "get_current_user",
    "get_current_active_user",
    "check_user_permissions",

    # Models
    "Token",
    "User",
    "UserInDB",
    "TokenData",

    # Middleware and dependencies
    "get_current_user_dependency",
    "require_permission",
    "require_classification_read",
    "require_classification_write",
    "require_evaluation_read",
    "require_evaluation_write",
    "require_admin_read",
    "require_admin_write",
    "AuthMiddleware"
]
