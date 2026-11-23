# reto-20

Agente comunitario analítico basado en RAG para consultar documentos de una organización y producir respuestas con referencias.

## Características

- Carga de archivos `.txt` y `.md` mediante API o desde la carpeta `data/`.
- Fragmentación de documentos y búsqueda con TF-IDF.
- Respuesta extractiva basada en los fragmentos más relevantes.
- Interfaz web sencilla para preguntas y carga de documentos.
- API con documentación interactiva en `/docs`.
- Persistencia opcional de documentos en memoria durante la ejecución.

## Inicio rápido

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Abre <http://localhost:8000>.

## Endpoints

- `GET /health`: estado del servicio.
- `GET /documents`: documentos cargados.
- `POST /documents`: carga un archivo `.txt` o `.md`.
- `POST /ask`: consulta al agente.

Ejemplo:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"¿Cuál es la política principal?"}'
```

## Alcance

El MVP no necesita una clave externa de LLM: recupera los fragmentos relevantes y construye una respuesta fundamentada a partir de ellos. Después puede integrarse un proveedor generativo opcional.
