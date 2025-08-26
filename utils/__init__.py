"""
Utilidades auxiliares para la API médica
"""
from pathlib import Path


def ensure_directory(path: str | Path) -> None:
    """Asegurar que un directorio existe"""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """Obtener la raíz del proyecto"""
    return Path(__file__).parent.parent


def setup_directories():
    """Configurar directorios necesarios para la aplicación"""
    root = get_project_root()

    directories = [
        root / "logs",
        root / "data" / "temp",
        root / "data" / "processed",
        root / "data" / "cache"
    ]

    for directory in directories:
        ensure_directory(directory)


if __name__ == "__main__":
    setup_directories()
    print("✅ Directorios configurados correctamente")
