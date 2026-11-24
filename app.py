from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CHUNK_SIZE = 420
CHUNK_OVERLAP = 70


@dataclass
class Chunk:
    document: str
    text: str


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class RAGAgent:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.documents: dict[str, str] = {}
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None

    @staticmethod
    def split_text(text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(text):
                break
            start = end - CHUNK_OVERLAP
        return chunks

    def add_document(self, name: str, content: str) -> int:
        content = content.strip()
        if not content:
            raise ValueError("El documento está vacío")
        if Path(name).suffix.lower() not in {".txt", ".md"}:
            raise ValueError("Solo se permiten archivos .txt y .md")

        self.documents[name] = content
        pieces = self.split_text(content)
        self.chunks = [chunk for chunk in self.chunks if chunk.document != name]
        self.chunks.extend(Chunk(name, piece) for piece in pieces)
        self.rebuild_index()
        return len(pieces)

    def rebuild_index(self) -> None:
        if not self.chunks:
            self.vectorizer = None
            self.matrix = None
            return
        texts = [chunk.text for chunk in self.chunks]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=None,
            ngram_range=(1, 2),
        )
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, question: str, limit: int = 3) -> list[tuple[float, Chunk]]:
        if self.vectorizer is None or self.matrix is None:
            return []
        query = self.vectorizer.transform([question])
        scores = cosine_similarity(query, self.matrix).flatten()
        positions = scores.argsort()[::-1][:limit]
        return [(float(scores[index]), self.chunks[index]) for index in positions]

    def ask(self, question: str) -> dict:
        matches = self.search(question)
        matches = [(score, chunk) for score, chunk in matches if score > 0]
        if not matches:
            return {
                "answer": "No encontré información relevante en los documentos cargados. Intenta reformular la pregunta o cargar documentación relacionada.",
                "sources": [],
                "confidence": 0,
            }

        sources = []
        excerpts = []
        seen_documents = set()
        for score, chunk in matches:
            excerpt = chunk.text if len(chunk.text) <= 420 else chunk.text[:420].rstrip() + "..."
            excerpt = re.sub(r"\s+", " ", excerpt)
            excerpts.append(f"• {excerpt}")
            if chunk.document not in seen_documents:
                sources.append({"document": chunk.document, "score": round(score, 3)})
                seen_documents.add(chunk.document)

        confidence = min(1.0, matches[0][0])
        answer = (
            "Según la documentación disponible, la información más relevante es:\n\n"
            + "\n".join(excerpts)
        )
        return {
            "answer": answer,
            "sources": sources,
            "confidence": round(confidence, 3),
        }


agent = RAGAgent()

for path in sorted(DATA_DIR.iterdir()) if DATA_DIR.exists() else []:
    if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
        agent.add_document(path.name, path.read_text(encoding="utf-8"))

app = FastAPI(title="Agente comunitario analítico", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(agent.documents), "chunks": len(agent.chunks)}


@app.get("/documents")
def documents():
    return {
        "documents": [
            {"name": name, "characters": len(content)}
            for name, content in sorted(agent.documents.items())
        ]
    }


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="El archivo no tiene nombre")
    if Path(file.filename).suffix.lower() not in {".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Solo se permiten archivos .txt y .md")

    content = (await file.read()).decode("utf-8")
    try:
        chunk_count = agent.add_document(file.filename, content)
        saved = DATA_DIR / Path(file.filename).name
        saved.write_text(content, encoding="utf-8")
    except (UnicodeDecodeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {"name": file.filename, "chunks": chunk_count}


@app.post("/ask")
def ask(payload: Question):
    return agent.ask(payload.question)


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agente comunitario analítico</title>
  <style>
    body{font-family:system-ui;max-width:860px;margin:40px auto;padding:0 20px;background:#f6f7fb;color:#172033}
    h1{margin-bottom:8px}.card{background:white;border-radius:12px;padding:20px;margin:18px 0;box-shadow:0 3px 14px #cbd2e1}
    textarea,input,button{font:inherit}
    textarea{width:100%;min-height:90px;padding:10px;box-sizing:border-box}
    button{padding:10px 16px;border:0;border-radius:8px;background:#4255bd;color:white;cursor:pointer}
    button:disabled{cursor:wait;opacity:.7}
    pre{white-space:pre-wrap}small{color:#68738a}
  </style>
</head>
<body>
  <h1>Agente comunitario analítico</h1>
  <small>Consulta fundamentada sobre documentos comunitarios</small>

  <div class="card">
    <h2>Cargar documento</h2>
    <input id="file" type="file" accept=".txt,.md">
    <button type="button" id="uploadButton">Cargar</button>
    <p id="uploadStatus"></p>
  </div>

  <div class="card">
    <h2>Realizar consulta</h2>
    <textarea id="question" placeholder="Ejemplo: ¿Cómo solicitar apoyo comunitario?"></textarea><br><br>
    <button type="button" id="askButton">Consultar</button>
    <pre id="answer"></pre>
  </div>

<script>
async function uploadDocument() {
  const file = document.getElementById('file').files[0];
  const status = document.getElementById('uploadStatus');

  if (!file) {
    status.textContent = 'Selecciona primero un archivo .txt o .md.';
    return;
  }

  const form = new FormData();
  form.append('file', file);

  try {
    const response = await fetch('/documents', { method: 'POST', body: form });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || 'Error al cargar el documento');
    }

    status.textContent = 'Documento cargado: ' + data.name;
  } catch (error) {
    status.textContent = 'Error: ' + error.message;
  }
}

async function askQuestion() {
  const question = document.getElementById('question').value.trim();
  const answerElement = document.getElementById('answer');
  const button = document.getElementById('askButton');

  if (question.length < 3) {
    answerElement.textContent = 'Escribe una pregunta de al menos tres caracteres.';
    return;
  }

  button.disabled = true;
  answerElement.textContent = 'Consultando...';

  try {
    const response = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: question })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || 'Error al realizar la consulta');
    }

    answerElement.textContent = data.answer
      + '\n\nFuentes: '
      + JSON.stringify(data.sources, null, 2);
  } catch (error) {
    answerElement.textContent = 'Error: ' + error.message;
  } finally {
    button.disabled = false;
  }
}

document.getElementById('uploadButton').addEventListener('click', uploadDocument);
document.getElementById('askButton').addEventListener('click', askQuestion);
</script>
</body>
</html>"""
