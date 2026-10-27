"""Relant RAG (línea de comandos): retriever híbrido (BM25 + Chroma, RRF, reranker
cross-encoder) + router fast-path / agente LangGraph. Sin interfaz web, sin usuarios,
sin cotizaciones y sin importaciones. Uso: python -m app.cli --help"""

APP_NAME = "Relant RAG CLI"
__version__ = "1.0.0-cli"
