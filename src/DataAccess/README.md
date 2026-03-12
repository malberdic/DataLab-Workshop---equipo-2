# Data Access Layer (DAL) 🗄️

## Descripción
Esta capa es la responsable de la gestión integral de los datos. Se encarga desde la extracción de reseñas desde fuentes externas hasta su refinamiento y persistencia final. Centraliza toda la interacción con sistemas de almacenamiento y proveedores de datos.

## Estructura y Subcapas

### 1. Ingestion (Extracción)
Es el punto de contacto con el mundo exterior para la obtención de datos crudos.
- **Responsabilidad:** Conectarse a APIs de terceros (App Store, Google Play, etc.) o servicios de scraping.
- **Contenido:** Conectores específicos por plataforma, manejo de paginación de reseñas y control de límites de cuota (Rate Limiting).
- **Salida:** Datos en formato crudo (Raw Data).

### 2. Refinery (Procesamiento)
Transforma los datos crudos en información apta para ser procesada por un LLM.
- **Responsabilidad:** Limpieza, normalización y curado.
- **Tareas clave:** 
    - Eliminación de ruido (caracteres especiales, HTML, spam).
    - Detección y filtrado de PII (Información de Identificación Personal).
    - Traducción o detección de idioma si es necesario.
    - Fragmentación (Chunking) estratégica para optimizar la ventana de contexto. **(a consultar)**
- **Salida:** Datos limpios y estructurados.

### 3. Storage (Persistencia)
Gestiona el ciclo de vida de los datos en reposo.
- **Responsabilidad:** Lectura y escritura en sistemas de bases de datos.
- **Componentes:**
    - **Raw Storage:** Persistencia de reseñas originales para auditoría.
    - **Processed Storage:** Almacenamiento de datos listos para inferencia.
    - **Result Storage:** Guardado de los análisis FODA generados y metadatos asociados.
- **Tecnologías:** Implementaciones de bases de datos relacionales (SQL), NoSQL o Vectoriales según la necesidad.

## Interacciones
- **Hacia arriba:** Provee métodos limpios a la capa de **Orchestration** (ej. `get_cleaned_reviews()`).
- **Aislamiento:** La lógica de "cómo" se obtiene o guarda un dato está encapsulada aquí; el resto de la app no sabe si los datos vienen de una API o de un archivo local.

---
*Nota: Se recomienda seguir el patrón Repository para desacoplar la lógica de acceso a datos de la infraestructura subyacente.*
