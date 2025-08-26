"""
Script de demostración de la API de Clasificación Médica
"""
import json
import time
from datetime import datetime
from typing import Any

import requests

# Configuración de la API
API_BASE_URL = "http://localhost:8000"
API_ENDPOINTS = {
    "login": f"{API_BASE_URL}/api/v1/auth/login",
    "demo_users": f"{API_BASE_URL}/api/v1/auth/users/demo",
    "single_classification": f"{API_BASE_URL}/api/v1/classification/single",
    "batch_classification": f"{API_BASE_URL}/api/v1/classification/batch",
    "quick_test": f"{API_BASE_URL}/api/v1/classification/quick-test",
    "methods": f"{API_BASE_URL}/api/v1/classification/methods",
    "domains": f"{API_BASE_URL}/api/v1/classification/domains",
    "health": f"{API_BASE_URL}/health",
    "info": f"{API_BASE_URL}/api/v1/info"
}


class MedicalAPIDemo:
    """Cliente de demostración para la API médica"""

    def __init__(self):
        self.session = requests.Session()
        self.token = None
        self.headers = {}

    def print_section(self, title: str):
        """Imprimir separador de sección"""
        print("\n" + "="*60)
        print(f"🧪 {title}")
        print("="*60)

    def print_result(self, title: str, data: dict[str, Any]):
        """Imprimir resultado de manera formateada"""
        print(f"\n📊 {title}:")
        print(json.dumps(data, indent=2, ensure_ascii=False))

    def login(self, username: str = "admin", password: str = "secret") -> bool:
        """Autenticarse en la API"""
        try:
            login_data = {
                "username": username,
                "password": password
            }

            response = self.session.post(
                API_ENDPOINTS["login"],
                data=login_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )

            if response.status_code == 200:
                token_data = response.json()
                self.token = token_data["access_token"]
                self.headers = {"Authorization": f"Bearer {self.token}"}
                print(f"✅ Login exitoso como {username}")
                self.print_result("Token Info", {
                    "user": token_data.get("user_info", {}),
                    "expires_in": f"{token_data.get('expires_in', 0)}s",
                    "token_type": token_data.get("token_type", "")
                })
                return True
            else:
                print(f"❌ Error en login: {response.status_code}")
                print(response.text)
                return False

        except Exception as e:
            print(f"❌ Error de conexión: {e}")
            return False

    def test_health_endpoints(self):
        """Probar endpoints de salud e información"""
        self.print_section("HEALTH & INFO ENDPOINTS")

        # Health check
        try:
            response = self.session.get(API_ENDPOINTS["health"])
            self.print_result("Health Check", response.json())
        except Exception as e:
            print(f"❌ Error en health check: {e}")

        # API Info
        try:
            response = self.session.get(API_ENDPOINTS["info"])
            self.print_result("API Info", response.json())
        except Exception as e:
            print(f"❌ Error en API info: {e}")

    def test_authentication(self):
        """Probar sistema de autenticación"""
        self.print_section("AUTHENTICATION SYSTEM")

        # Ver usuarios demo
        try:
            response = self.session.get(API_ENDPOINTS["demo_users"])
            self.print_result("Demo Users", response.json())
        except Exception as e:
            print(f"❌ Error obteniendo usuarios demo: {e}")

        # Login como admin
        return self.login("admin", "secret")

    def test_classification_info(self):
        """Probar endpoints de información de clasificación"""
        self.print_section("CLASSIFICATION INFO")

        # Métodos disponibles
        try:
            response = self.session.get(API_ENDPOINTS["methods"], headers=self.headers)
            self.print_result("Available Methods", response.json())
        except Exception as e:
            print(f"❌ Error obteniendo métodos: {e}")

        # Dominios médicos
        try:
            response = self.session.get(API_ENDPOINTS["domains"], headers=self.headers)
            self.print_result("Medical Domains", response.json())
        except Exception as e:
            print(f"❌ Error obteniendo dominios: {e}")

    def test_quick_classification(self):
        """Probar clasificación rápida"""
        self.print_section("QUICK CLASSIFICATION TEST")

        test_cases = [
            {
                "text": "Patient with acute myocardial infarction treated with stent placement",
                "method": "biobert"
            },
            {
                "text": "Study on alzheimer disease progression and cognitive decline in elderly patients",
                "method": "hybrid"
            },
            {
                "text": "Liver cirrhosis treatment with new hepatoprotective drugs",
                "method": "biobert"
            }
        ]

        for i, test_case in enumerate(test_cases, 1):
            try:
                print(f"\n🧪 Test Case {i}: {test_case['method'].upper()}")
                response = self.session.post(
                    API_ENDPOINTS["quick_test"],
                    json=test_case,
                    headers=self.headers
                )

                if response.status_code == 200:
                    result = response.json()
                    print(f"📝 Text: {test_case['text'][:50]}...")
                    print(f"🎯 Domains: {result.get('predicted_domains', [])}")
                    print(f"⏱️  Time: {result.get('processing_time', 0):.3f}s")
                    print(f"🔧 Method: {result.get('method_used', 'unknown')}")
                else:
                    print(f"❌ Error {response.status_code}: {response.text}")

            except Exception as e:
                print(f"❌ Error en test case {i}: {e}")

    def test_single_classification(self):
        """Probar clasificación de artículo individual"""
        self.print_section("SINGLE ARTICLE CLASSIFICATION")

        test_articles = [
            {
                "title": "Effects of ACE inhibitors on cardiovascular outcomes in hypertensive patients",
                "abstract": "This randomized controlled trial evaluates the cardiovascular benefits of ACE inhibitors in patients with essential hypertension. Results show significant reduction in myocardial infarction and stroke rates.",
                "authors": "Smith J., Johnson A., Brown K.",
                "journal": "Journal of Cardiology"
            },
            {
                "title": "Novel therapeutic approaches for Alzheimer's disease",
                "abstract": "Recent advances in understanding amyloid-beta pathology have led to development of new therapeutic strategies for Alzheimer's disease. This review discusses current clinical trials and future directions.",
                "keywords": ["alzheimer", "amyloid", "neurodegeneration", "therapy"]
            }
        ]

        for i, article in enumerate(test_articles, 1):
            try:
                print(f"\n📄 Article {i}: {article['title'][:50]}...")

                response = self.session.post(
                    API_ENDPOINTS["single_classification"],
                    json=article,
                    headers=self.headers,
                    params={"method": "hybrid"}
                )

                if response.status_code == 200:
                    result = response.json()
                    print(f"🎯 Predicted Domains: {result.get('domains', [])}")
                    print("📊 Confidence Scores:")
                    for domain, score in result.get('confidence_scores', {}).items():
                        print(f"   {domain}: {score:.3f}")
                    print(f"🔧 Method Used: {result.get('method_used', 'unknown')}")
                    print(f"⏱️  Processing Time: {result.get('processing_time', 0):.3f}s")
                else:
                    print(f"❌ Error {response.status_code}: {response.text}")

            except Exception as e:
                print(f"❌ Error en artículo {i}: {e}")

    def test_batch_classification(self):
        """Probar clasificación en lote"""
        self.print_section("BATCH CLASSIFICATION")

        batch_articles = [
            {
                "title": "Cardiac surgery outcomes in elderly patients",
                "abstract": "Analysis of cardiac surgery outcomes and complications in patients over 75 years old."
            },
            {
                "title": "Brain tumor classification using MRI imaging",
                "abstract": "Machine learning approaches for automated brain tumor classification from MRI scans."
            },
            {
                "title": "Breast cancer chemotherapy protocols",
                "abstract": "Comparison of different chemotherapy regimens for breast cancer treatment."
            },
            {
                "title": "Kidney transplant rejection mechanisms",
                "abstract": "Study of immunological mechanisms leading to kidney transplant rejection."
            }
        ]

        batch_request = {
            "articles": batch_articles,
            "method": "hybrid",
            "confidence_threshold": 0.7,
            "parallel_processing": True
        }

        try:
            print(f"📦 Processing {len(batch_articles)} articles in batch...")
            start_time = time.time()

            response = self.session.post(
                API_ENDPOINTS["batch_classification"],
                json=batch_request,
                headers=self.headers
            )

            total_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()

                print(f"✅ Batch processing completed in {total_time:.3f}s")
                print("📊 Summary:")
                print(f"   Total Processed: {result.get('total_processed', 0)}")
                print(f"   Total Time: {result.get('total_time', 0):.3f}s")
                print(f"   Success Rate: {result.get('summary', {}).get('success_rate', 0):.1f}%")

                # Mostrar resultados por artículo
                results = result.get('results', [])
                for i, article_result in enumerate(results[:3]):  # Mostrar solo primeros 3
                    print(f"\n📄 Article {i+1} Results:")
                    print(f"   Domains: {article_result.get('domains', [])}")
                    print(f"   Method: {article_result.get('method_used', 'unknown')}")
                    print(f"   Time: {article_result.get('processing_time', 0):.3f}s")

                if len(results) > 3:
                    print(f"\n... y {len(results) - 3} artículos más")

            else:
                print(f"❌ Error {response.status_code}: {response.text}")

        except Exception as e:
            print(f"❌ Error en clasificación en lote: {e}")

    def run_full_demo(self):
        """Ejecutar demostración completa"""
        print("🏥 MEDICAL CLASSIFICATION API - DEMO COMPLETO")
        print(f"⏰ Timestamp: {datetime.now().isoformat()}")
        print(f"🌐 Base URL: {API_BASE_URL}")

        # 1. Health & Info
        self.test_health_endpoints()

        # 2. Authentication
        if not self.test_authentication():
            print("❌ No se pudo autenticar, abortando demo...")
            return

        # 3. Classification Info
        self.test_classification_info()

        # 4. Quick Tests
        self.test_quick_classification()

        # 5. Single Classification
        self.test_single_classification()

        # 6. Batch Classification
        self.test_batch_classification()

        print("\n" + "="*60)
        print("🎉 DEMO COMPLETADO EXITOSAMENTE")
        print("="*60)
        print("\n📚 Para más información:")
        print(f"   - Documentación: {API_BASE_URL}/docs")
        print(f"   - ReDoc: {API_BASE_URL}/redoc")
        print(f"   - OpenAPI: {API_BASE_URL}/openapi.json")


def main():
    """Función principal del demo"""
    demo = MedicalAPIDemo()

    print("🚀 Iniciando demostración de la API...")
    print("⚠️  Asegúrate de que la API esté ejecutándose en http://localhost:8000")
    input("Presiona Enter para continuar...")

    demo.run_full_demo()


if __name__ == "__main__":
    main()
