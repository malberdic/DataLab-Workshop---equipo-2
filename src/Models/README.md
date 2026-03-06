# Models Layer 🤖

## Descripción
Esta capa gestiona la comunicación con los **Large Language Models (LLMs)** y otros modelos de Machine Learning. Su objetivo es abstraer la complejidad de las APIs de inferencia y proporcionar una interfaz uniforme para el resto de la aplicación.

## Responsabilidades
- **Abstracción de Proveedores:** Actúa como un *wrapper* para diferentes servicios (OpenAI, Anthropic, Google Gemini) o modelos locales (Ollama, vLLM).
- **Gestión de Inferencia:** Configura los parámetros del modelo (temperatura, max_tokens, stop sequences) para garantizar resultados predecibles.
- **Manejo de Prompt Templates:** Almacena y versiona las plantillas de instrucciones, como la estructura necesaria para generar un análisis **FODA**.
- **Normalización de Salida:** Asegura que la respuesta del modelo (a menudo texto libre) se transforme en el formato estructurado (JSON) que la aplicación espera.

## Estructura de Contenido
- `providers/`: Implementaciones específicas para cada cliente de IA (ej. `openai_client.py`, `anthropic_client.py`).
- `templates/`: Diccionario o archivos de prompts organizados por caso de uso (ej. `foda_prompt_v1`).
- `configs/`: Parámetros de configuración por modelo (model_id, límites de contexto).
- `formatters/`: Lógica para parsear y validar la salida del modelo (ej. uso de LangChain Output Parsers).

## Interacciones
- **Hacia arriba:** Recibe instrucciones y datos procesados de la **Orchestration Layer**.
- **Hacia afuera:** Realiza llamadas de red a las APIs de los modelos de lenguaje.
- **Aislamiento:** Ninguna otra capa debe conocer qué modelo específico se está utilizando; solo interactúan con la interfaz definida en esta capa.

---
*Nota: Es crítico implementar aquí mecanismos de "Fallback" (si falla GPT-4, intentar con otro modelo) para asegurar la resiliencia del análisis.*
