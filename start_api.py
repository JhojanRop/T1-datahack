"""
Script de inicio rápido para la API de Clasificación Médica
"""
import shutil
import subprocess
import sys
from pathlib import Path


def print_header():
    """Imprimir header del script"""
    print("🏥" + "="*58 + "🏥")
    print("   MEDICAL CLASSIFICATION API - STARTUP SCRIPT")
    print("🏥" + "="*58 + "🏥")


def check_python_version():
    """Verificar versión de Python"""
    print("\n🐍 Verificando versión de Python...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Se requiere Python 3.8 o superior")
        print(f"   Versión actual: {version.major}.{version.minor}")
        return False

    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    return True


def check_model_exists():
    """Verificar que existe el modelo BioBERT"""
    print("\n🧠 Verificando modelo BioBERT...")
    model_path = Path("./model/biobert_finetuned_v3")

    if not model_path.exists():
        print("⚠️  Modelo BioBERT no encontrado")
        print(f"   Ruta esperada: {model_path.absolute()}")
        print("   El sistema funcionará en modo simulación")
        return False

    required_files = ["config.json", "model.safetensors", "tokenizer.json"]
    missing_files = []

    for file in required_files:
        if not (model_path / file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"⚠️  Archivos faltantes del modelo: {missing_files}")
        print("   El sistema funcionará en modo simulación")
        return False

    print("✅ Modelo BioBERT encontrado - OK")
    return True


def setup_environment():
    """Configurar entorno"""
    print("\n🔧 Configurando entorno...")

    # Copiar .env.example a .env si no existe
    env_file = Path(".env")
    env_example = Path(".env.example")

    if not env_file.exists() and env_example.exists():
        shutil.copy(env_example, env_file)
        print("✅ Archivo .env creado desde .env.example")
    elif env_file.exists():
        print("✅ Archivo .env ya existe")
    else:
        print("⚠️  No se encontró .env.example")

    # Crear directorios necesarios
    directories = ["logs", "data/temp", "data/processed", "data/cache"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("✅ Directorios creados")


def install_dependencies():
    """Instalar dependencias"""
    print("\n📦 Verificando dependencias...")

    # Verificar si existe pyproject.toml
    if not Path("pyproject.toml").exists():
        print("⚠️  pyproject.toml no encontrado")
        print("   Instalación manual de dependencias requerida")
        return False

    try:
        # Intentar instalar con uv si está disponible
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ UV package manager encontrado")
            print("🔄 Instalando dependencias con UV...")
            subprocess.run(["uv", "sync"], check=True)
            print("✅ Dependencias instaladas con UV")
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    try:
        # Fallback a pip
        print("🔄 Instalando dependencias con pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
        print("✅ Dependencias instaladas con pip")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False


def start_api():
    """Iniciar la API"""
    print("\n🚀 Iniciando API de Clasificación Médica...")
    print("   Base URL: http://localhost:8000")
    print("   Documentación: http://localhost:8000/docs")
    print("   Health Check: http://localhost:8000/health")
    print("\n⏹️  Presiona Ctrl+C para detener el servidor")
    print("=" * 60)

    try:
        # Intentar con uvicorn directamente
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "api_main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ], check=True)
    except subprocess.CalledProcessError:
        # Fallback a ejecución directa
        try:
            subprocess.run([sys.executable, "api_main.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error iniciando la API: {e}")
            print("🔧 Intenta ejecutar manualmente:")
            print("   python api_main.py")
            print("   O: uvicorn api_main:app --reload")


def run_demo():
    """Ejecutar demostración"""
    print("\n🧪 ¿Deseas ejecutar la demostración de la API?")
    choice = input("   (y/n): ").lower().strip()

    if choice in ['y', 'yes', 'sí', 's']:
        try:
            subprocess.run([sys.executable, "demo_api.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error ejecutando demo: {e}")
        except KeyboardInterrupt:
            print("\n⏹️  Demo interrumpido por el usuario")


def main():
    """Función principal"""
    print_header()

    # Verificaciones previas
    if not check_python_version():
        sys.exit(1)

    model_available = check_model_exists()

    # Configuración
    setup_environment()

    # Dependencias
    if not install_dependencies():
        print("\n⚠️  Continuando sin instalar dependencias...")
        print("   Asegúrate de instalar manualmente:")
        print("   pip install fastapi uvicorn transformers torch pydantic")

    # Información adicional
    print("\n📋 INFORMACIÓN IMPORTANTE:")
    print("   • La API estará disponible en http://localhost:8000")
    print("   • Documentación interactiva en /docs")
    print("   • Usuarios demo: admin/secret, researcher/secret, viewer/secret")

    if not model_available:
        print("   • ⚠️  Modelo BioBERT no disponible - funcionará en modo simulación")

    print("   • Para obtener API key de Gemini: https://makersuite.google.com/app/apikey")

    # Preguntar si iniciar la API
    print("\n🚀 ¿Deseas iniciar la API ahora?")
    choice = input("   (y/n): ").lower().strip()

    if choice in ['y', 'yes', 'sí', 's']:
        start_api()
    else:
        print("\n✅ Configuración completada")
        print("🔧 Para iniciar manualmente:")
        print("   python api_main.py")
        print("   O: uvicorn api_main:app --reload")

        # Opción de demo
        run_demo()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Script interrumpido por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        sys.exit(1)
