"""Settings del RAG (versión de línea de comandos: sin interfaz web, sin usuarios,
sin cotizaciones y sin importaciones).

TODAS las rutas vienen del .env (ninguna está escrita en el código):

* El .env que se lee es UNO solo: el de la variable de entorno RAG_ENV_FILE si existe,
  si no el `.env` de la raíz del proyecto (junto a la carpeta app/). No depende de la
  carpeta desde donde se ejecuta el comando.
* Las rutas relativas del .env se resuelven contra RAG_BASE_DIR (si no se define, la
  carpeta donde está el .env), nunca contra el directorio de trabajo.
* Las carpetas de cada tipo de documento (productos, hojas técnicas, manuales, dibujos...)
  se declaran en el .env como RAG_DIR_* (relativas a RAG_DATA_DIR). Esa declaración es la
  que decide qué es cada archivo: un PDF dentro de RAG_DIR_MANUALES es un manual aunque su
  nombre diga "ficha técnica", y sólo RAG_DIR_HOJAS_TECNICAS cuenta como hoja técnica.
* Si falta una variable de ruta obligatoria, la app no arranca y dice cuál falta.

"""
from __future__ import annotations

import functools
import os
from pathlib import Path

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Carpeta del código (no es una ruta de datos): sólo sirve para encontrar el .env por defecto.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = Path(os.environ.get("RAG_ENV_FILE") or (PROJECT_ROOT / ".env")).expanduser()

# ---- file types (fixed, not settings) -------------------------------------------------
PDF_EXTS = (".pdf",)
TEXT_EXTS = (".txt", ".md")
CSV_EXTS = (".csv",)
EXCEL_EXTS = (".xlsx", ".xlsm", ".xls")
TABLE_EXTS = CSV_EXTS + EXCEL_EXTS     # tablas: CSV y Excel se leen igual (textutils.read_csv_table)
IMAGE_EXTS = (".png", ".jpg", ".jpeg")

# display name, keys searched in file path / first page (catalog)
MANUFACTURERS = (
    ("ABNOX", ("abnox",)),
    ("Schütze", ("schutze", "alfred schutze")),
    ("Soma", ("soma",)),
    ("Lubtec", ("lubtec",)),
    ("Walther Systemtechnik", ("walther",)),
)

# =========================================================================================
# Secciones de la carpeta de datos. La CARPETA de cada sección viene del .env (RAG_DIR_*);
# aquí sólo está qué significa cada sección.
# =========================================================================================
SECTION_ENV = {                       # sección -> variable del .env
    "productos": "RAG_DIR_PRODUCTOS",
    "precios": "RAG_DIR_PRECIOS",
    "fotos": "RAG_DIR_FOTOS",
    "explosionados": "RAG_DIR_EXPLOSIONADOS",
    "drawings": "RAG_DIR_DRAWINGS",
    "hojas_tecnicas": "RAG_DIR_HOJAS_TECNICAS",
    "manuales": "RAG_DIR_MANUALES",
    "catalogos": "RAG_DIR_CATALOGOS",
}
OTHER_SECTION = "otros"               # todo lo que no está en ninguna carpeta declarada
SECTION_DOC_TYPE = {                  # tipo de documento del catálogo, fijado por la carpeta
    "productos": "table", "precios": "table",
    "fotos": "drawing", "explosionados": "drawing", "drawings": "drawing",
    "hojas_tecnicas": "datasheet", "manuales": "manual", "catalogos": "catalogue",
}
SECTION_LABEL = {                     # etiqueta que ve el LLM junto a cada fragmento
    "productos": "PRODUCT DATA (product CSV)", "precios": "PRICE/STOCK LIST",
    "hojas_tecnicas": "DATASHEET", "manuales": "MANUAL", "catalogos": "CATALOGUE",
    "fotos": "PRODUCT PHOTO", "explosionados": "EXPLODED VIEW", "drawings": "DRAWING",
    OTHER_SECTION: "OTHER DOCUMENT",
}
IMAGE_SECTIONS = ("fotos", "explosionados", "drawings")   # sólo aquí se indexan imágenes
# Fuentes que el agente / el router pueden pedir (nombre -> secciones)
SOURCE_GROUPS = {
    "datasheet": ("productos", "hojas_tecnicas"),
    "manual": ("manuales",),
    "drawing": ("explosionados", "drawings", "fotos"),
    "catalogue": ("catalogos",),
    "price": ("precios",),
    "other": (OTHER_SECTION,),
}
SOURCE_ALIASES = {
    "datasheet": "datasheet", "datasheets": "datasheet", "data sheet": "datasheet",
    "hoja tecnica": "datasheet", "hojas tecnicas": "datasheet", "ficha tecnica": "datasheet",
    "spec": "datasheet", "specs": "datasheet", "product": "datasheet", "producto": "datasheet",
    "manual": "manual", "manuals": "manual", "manuales": "manual", "instructivo": "manual",
    "instructions": "manual",
    "drawing": "drawing", "drawings": "drawing", "dibujo": "drawing", "dibujos": "drawing",
    "exploded": "drawing", "explosionado": "drawing", "despiece": "drawing", "plano": "drawing",
    "image": "drawing", "foto": "drawing",
    "catalogue": "catalogue", "catalog": "catalogue", "catalogo": "catalogue",
    "catalogos": "catalogue",
    "price": "price", "prices": "price", "precio": "price", "precios": "price",
    "stock": "price", "existencia": "price", "existencias": "price",
    "other": "other", "otros": "other",
}

WEB_SITES = {
    "abnox": {"name": "ABNOX", "start": "https://www.abnox.com/en",
              "sitemap": "https://www.abnox.com/sitemap.xml", "prefix": "/en"},
    "schuetze": {"name": "Alfred Schütze Apparatebau",
                 "start": "https://www.schuetze-gmbh.de/en/homepage",
                 "sitemap": "https://www.schuetze-gmbh.de/sitemap.xml", "prefix": "/en"},
}
WEB_SKIP = ("/legal-notice", "/data-privacy", "/privacy", "/imprint", "/jobs", "/career",
            "/return-of-goods", "/customer-login", "/login", "/search", "/cart", "/account")


class ConfigError(RuntimeError):
    """Falta o es inválida una variable del .env."""


def _dir_list(raw: str | None) -> tuple[str, ...]:
    """'drawings/fotos, Fotos' -> ('drawings/fotos', 'fotos'): relative, '/' separated,
    case-insensitive. Absolute values are left as they are (checked by the validator)."""
    out = []
    for part in str(raw or "").replace(";", ",").split(","):
        p = part.strip().strip('"').strip("'").replace("\\", "/")
        if not p:
            continue
        if not (p.startswith("/") or (len(p) > 1 and p[1] == ":")):
            p = p.strip("/")
            while p.startswith("./"):
                p = p[2:]
        out.append(p)
    return tuple(out)


@functools.lru_cache(maxsize=8)
def _section_prefixes(spec: tuple[tuple[str, tuple[str, ...]], ...]) -> tuple:
    """((('drawings','fotos'), 'fotos'), (('drawings',), 'drawings'), ...) longest first,
    so 'drawings/fotos' wins over 'drawings'."""
    pref = []
    for section, dirs in spec:
        for d in dirs:
            segs = tuple(s.casefold() for s in d.split("/") if s and s != ".")
            if segs:
                pref.append((segs, section))
    pref.sort(key=lambda x: len(x[0]), reverse=True)
    return tuple(pref)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ENV_FILE), env_file_encoding="utf-8-sig",
                                      extra="ignore", populate_by_name=True)

    # ---- OpenRouter ----
    openrouter_api_key: str = Field(
        "", validation_alias=AliasChoices("OPENROUTER_API_KEY_RELANT", "OPENROUTER_API_KEY"))
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openai/gpt-5.4-mini"        # fast path + final answers
    agent_model: str = ""                         # agent decisions ("" -> llm_model)
    vision_model: str = ""                        # drawing descriptions ("" -> llm_model)
    # Same model as 5.7 -> the vectors already in Chroma and in the cache stay valid.
    embed_model: str = "openai/text-embedding-3-small"
    temperature: float = 0.0
    llm_timeout: int = 90
    embed_timeout: int = 60
    max_retries: int = 3
    preflight: bool = True                        # check key + credits before indexing
    # USD per 1M tokens, only for the cost estimate returned with each answer.
    price_in_per_mtok: float = 0.25
    price_out_per_mtok: float = 2.00
    # optional extra instructions added to every prompt (persona, tone, format). Replaces the
    # per-role "prompt profiles" of the web version. The `--instrucciones` option overrides it.
    instructions: str = Field("", validation_alias=AliasChoices("RAG_INSTRUCTIONS"))
    # ---- multilingual search: glossary (free) | llm (+1 small call if not English) | off
    query_translate: str = "glossary"

    # =====================================================================================
    # RUTAS (todas desde el .env). Relativas -> contra RAG_BASE_DIR.
    # =====================================================================================
    base_dir: str = Field("", validation_alias=AliasChoices("RAG_BASE_DIR"))
    # carpeta de documentos de la "Base general" (obligatoria)
    data_dir: str | None = Field(None, validation_alias=AliasChoices(
        "RAG_DATA_DIR", "RAG_PDF_DIR", "DATA_DIR"))
    # índice de la "Base general": chroma/, manifest.json, state.json (obligatoria)
    index_dir: str | None = Field(None, validation_alias=AliasChoices(
        "RAG_DB_DIR", "RAG_INDEX_DIR", "INDEX_DIR"))
    # logo de las hojas técnicas (opcional; por defecto assets/logo_relant.png)
    datasheet_logo: str = Field("", validation_alias=AliasChoices("RAG_DATASHEET_LOGO"))
    # carpetas (relativas a RAG_DATA_DIR, separadas por coma) que NUNCA se indexan,
    # p. ej. la carpeta de importaciones / invoices: RAG_EXCLUDE_DIRS=Importaciones
    exclude_dirs: str = Field("", validation_alias=AliasChoices("RAG_EXCLUDE_DIRS"))
    # opcionales: si se dejan vacías se derivan de RAG_DB_DIR
    chroma_dir_env: str = Field("", validation_alias=AliasChoices("RAG_CHROMA_DIR"))
    web_cache_dir_env: str = Field("", validation_alias=AliasChoices("RAG_WEB_CACHE_DIR"))
    datasheet_out_dir_env: str = Field("", validation_alias=AliasChoices("RAG_DATASHEET_OUT_DIR"))
    # opcional: config.json de app/pdfdatagen.py (plantilla HTML + colores por categoría
    # de las fichas técnicas PDF). Vacía = plantilla y colores por defecto.
    datasheet_pdf_config: str = Field("", validation_alias=AliasChoices("RAG_DATASHEET_PDF_CONFIG"))
    log_dir: str = Field("", validation_alias=AliasChoices("RAG_LOG_DIR"))   # "" = sólo consola

    # ---- secciones dentro de RAG_DATA_DIR ----
    # Relativas a la carpeta de datos; varias separadas por coma. Vacía = sección no usada.
    # Deben estar DECLARADAS en el .env (aunque sea vacías): si faltan, la app no arranca.
    dir_productos: str | None = Field(None, validation_alias=AliasChoices("RAG_DIR_PRODUCTOS"))
    dir_precios: str | None = Field(None, validation_alias=AliasChoices("RAG_DIR_PRECIOS"))
    dir_fotos: str | None = Field(None, validation_alias=AliasChoices("RAG_DIR_FOTOS"))
    dir_explosionados: str | None = Field(None, validation_alias=AliasChoices("RAG_DIR_EXPLOSIONADOS"))
    dir_drawings: str | None = Field(None, validation_alias=AliasChoices("RAG_DIR_DRAWINGS"))
    dir_hojas_tecnicas: str | None = Field(None, validation_alias=AliasChoices("RAG_DIR_HOJAS_TECNICAS"))
    dir_manuales: str | None = Field(None, validation_alias=AliasChoices("RAG_DIR_MANUALES"))
    dir_catalogos: str | None = Field(None, validation_alias=AliasChoices("RAG_DIR_CATALOGOS"))

    recursive: bool = True

    # ---- scripts/drive_sync.py (opcional): carpeta compartida de Google Drive ----
    drive_folder_url: str = ""
    drive_target_subdir: str = "drive"

    # ---- separación de fuentes (manuales vs hojas técnicas vs dibujos) ----
    # prefer = busca primero en la fuente que corresponde a la pregunta y sólo si ahí no hay
    #          nada usa el resto (etiquetado); strict = nunca sale de esa fuente;
    #          off = búsqueda mezclada como antes.
    source_routing: str = "prefer"
    # palabras (sin acentos, minúsculas) que indican de qué fuente debe salir la respuesta
    # palabras / frases completas (sin acentos) que indican de qué fuente sale la respuesta.
    # Se pueden reemplazar en el .env como lista JSON: INTENT_MANUAL_WORDS=["manual","falla"]
    intent_datasheet_words: tuple[str, ...] = (
        "especificacion", "especificaciones", "caracteristica", "caracteristicas",
        "datos tecnicos", "dato tecnico", "dimension", "dimensiones", "medida", "medidas",
        "peso", "presion", "presion maxima", "caudal", "conexion", "conexiones", "rosca",
        "material", "materiales", "temperatura", "voltaje", "tension", "potencia",
        "capacidad", "viscosidad", "rango", "rangos", "specification", "specifications",
        "specs", "pressure", "flow", "weight", "dimensions", "technical data")
    intent_manual_words: tuple[str, ...] = (
        "instalacion", "instalar", "mantenimiento", "operacion", "operar", "procedimiento",
        "paso a paso", "como se", "como hago", "como instalo", "falla", "fallas", "error",
        "errores", "alarma", "ajuste", "ajustar", "calibrar", "calibracion", "limpieza",
        "limpiar", "puesta en marcha", "arranque", "desmontar", "desmontaje", "montaje",
        "seguridad", "lubricacion", "troubleshooting", "maintenance", "installation",
        "operating", "how to")
    intent_drawing_words: tuple[str, ...] = (
        "posicion", "posiciones", "refaccion", "refacciones", "repuesto", "repuestos",
        "spare part", "spare parts", "pieza", "piezas", "foto", "fotos", "imagen",
        "imagenes", "ensamble")

    # ---- hybrid retriever ----
    top_k: int = 6
    # candidates taken from EACH list (vector, BM25) before fusion. 40 so the reranker
    # sees 30-40 fused candidates (was 20 in 5.7).
    candidate_pool: int = 40
    vector_weight: float = 0.4            # 5.7 min-max fusion only; unused since RRF (1.0)
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ---- Production 1.0: Reciprocal Rank Fusion (replaces min-max + damping) ----
    rrf_k: int = 60                       # score = sum_i  w_i / (rrf_k + rank_i)
    rrf_vector_weight: float = 1.0
    rrf_bm25_weight: float = 1.0

    # ---- Production 1.0: cross-encoder reranker (multilingual ES/EN/DE) ----
    # local  = sentence-transformers CrossEncoder on this server (model downloaded once)
    # cohere = Cohere Rerank API (COHERE_API_KEY);  off = fused order only
    reranker_backend: str = "local"
    # small + fast on CPU; for best quality (GPU or patient CPU): BAAI/bge-reranker-v2-m3
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_candidates: int = 40         # fused candidates re-scored; top_k (8-10) kept
    reranker_max_chars: int = 1600        # characters of each chunk sent to the reranker
    reranker_batch: int = 16
    reranker_device: str = ""             # "" = auto (cuda if available), "cpu", "cuda"
    cohere_api_key: str = Field("", validation_alias=AliasChoices("COHERE_API_KEY"))
    cohere_rerank_model: str = "rerank-v3.5"
    # chunks containing a part code that appears in the question are never pushed out of
    # the top-k by the reranker (cross-encoders are weak on codes like VDV-25)
    code_pinning: bool = True

    # ---- multilingual query (app/query_lang.py): QUERY_TRANSLATE is set above ----
    query_spell_fix: bool = True          # 'nozles' -> also search 'nozzles' (corpus words)

    # ---- Production 1.0: conversation context ----
    # auto = a follow-up ("¿y su caudal?", "what about the other one?") is rewritten into a
    #        standalone question from the last turns (1 small call to the agent model)
    # always = every question with history is rewritten;  off = each question alone (5.7)
    conversation_rewrite: str = "auto"
    conversation_history_turns: int = 3   # previous question/answer pairs used
    conversation_answer_chars: int = 700  # characters of each previous answer shown

    # ---- Production 1.0: BM25 analyzer ----
    # fold() accents/umlauts, Spanish + English stop words, Snowball stems (ES and EN),
    # plus the normalised part code (VDV-25 -> vdv25) as an extra token.
    bm25_stemming: bool = True

    # ---- Production 1.0: agent search tools ----
    max_queries_per_search: int = 2       # a search tool call may carry up to 2 queries
    max_query_chars: int = 300

    # ---- Production 1.0: free heuristic pre-route (no LLM call) ----
    preroute_enabled: bool = True

    # ---- indexing ----
    embed_api_batch: int = 100
    save_every: int = 10
    embed_flush_docs: int = 500
    chroma_batch: int = 5000
    vector_max_table_rows: int = 2000
    csv_rows_per_chunk: int = 1
    csv_forward_fill_cols: int = 0
    describe_images: bool = True
    vision_workers: int = 4
    image_max_side: int = 1600
    reload_check_secs: float = 10.0   # a long `chat` session picks up a new index this often

    # ---- document-level tools ----
    list_docs_max: int = 25
    read_doc_max_chars: int = 8000
    read_table_max_rows: int = 60
    table_lookup_max_rows: int = 15

    # ---- router (fast path vs agent) ----
    fast_top_k: int = 8                    # chunks sent to the single fast-path LLM call
    fast_min_vector_score: float = 0.35    # below this AND low term coverage -> escalate
    fast_min_term_coverage: float = 0.5    # share of query words found in the top chunks
    fast_escalate_on_insufficient: bool = True   # LLM says INSUFFICIENT_CONTEXT -> agent
    agent_max_steps: int = 6
    run_baseline: bool = False             # True only for evaluation (doubles the cost)
    max_drawings: int = 3

    # ---- caché semántica de RESPUESTAS (app/semantic_cache.py, SQLite) ----
    # Una pregunta ya respondida (igual o reformulada) se contesta desde la caché sin pagar
    # router + búsqueda + LLM. Ver CACHE_SEMANTICA.md.
    semantic_cache: bool = True
    semantic_cache_file_env: str = Field("", validation_alias=AliasChoices(
        "RAG_SEMANTIC_CACHE_FILE"))       # vacío = <RAG_DB_DIR>/semantic_cache.sqlite
    # coseno mínimo entre la pregunta nueva y una guardada (text-embedding-3-small).
    # ALTO a propósito: una respuesta equivocada desde caché es peor que un miss.
    semantic_cache_threshold: float = 0.88
    # un hit SEMÁNTICO (no el exacto) lo confirma un LLM barato antes de usarse
    semantic_cache_validate: bool = True
    semantic_cache_validator_model: str = ""      # "" = LLM_MODEL
    semantic_cache_ttl_days: float = 30.0         # 0 = sin caducidad
    semantic_cache_max_entries: int = 5000
    # sources = se descarta la respuesta si cambió/borró un archivo que usó (defecto)
    # index   = cualquier re-indexación vacía la caché;  off = sólo TTL
    semantic_cache_invalidation: str = "sources"
    # secciones cuyos datos cambian seguido: respuestas que las usan NO se guardan
    semantic_cache_skip_sections: tuple[str, ...] = ("precios",)

    # ---- log de trazas por pregunta (formato del Capstone 7.1: capstone_7_1_agent.log) ----
    trace_log: bool = True
    trace_log_file_env: str = Field("", validation_alias=AliasChoices(
        "RAG_TRACE_LOG"))                 # vacío = <RAG_LOG_DIR o ./logs>/relant_rag_agent.log

    # ---- manufacturer websites ----
    web_enabled: bool = True
    web_cache_days: int = 14
    web_max_pages: int = 150
    web_timeout: int = 15
    web_workers: int = 4

    # ---- product master data + technical datasheets (app/products.py) ----
    part_col_aliases: tuple[str, ...] = (
        "numero de parte", "no de parte", "num de parte", "n de parte", "numero parte",
        "part number", "part no", "codigo", "clave", "sku", "articulo",
        "no parte", "num parte", "modelo", "model")
    name_col_aliases: tuple[str, ...] = (
        "nombre del producto", "nombre producto", "producto", "nombre", "product name",
        "name", "descripcion", "description")
    family_col_aliases: tuple[str, ...] = ("familia", "family", "linea", "categoria")
    datasheet_hide_cols: tuple[str, ...] = (
        "costo", "coste", "cost", "precio", "price", "pvp", "existencia", "existencias",
        "stock", "inventario", "almacen", "ubicacion", "proveedor", "margen", "notas internas",
        "imagen", "image", "foto", "archivo")
    # An image is an exploded view if its folder or file name contains one of these
    # (las de RAG_DIR_EXPLOSIONADOS siempre lo son)
    exploded_words: tuple[str, ...] = ("explosionad", "explosion", "exploded", "despiece",
                                       "explotad")
    datasheet_image_max_side: int = 1200          # embedded images are resized to this
    product_cache_secs: float = 30.0
    # Folders whose CSVs are product families (one row per product). RAG_DIR_PRODUCTOS is the
    # documented place, but Relant keeps the family CSVs (Valvulas_apertura_y_cierre.csv,
    # valvulas_volumetricas.csv ...) inside RAG_DIR_HOJAS_TECNICAS, so both are read.
    # .env: PRODUCT_TABLE_SECTIONS='["productos","hojas_tecnicas"]'
    product_table_sections: tuple[str, ...] = ("productos", "hojas_tecnicas")

    # =====================================================================================
    # validation: every path variable declared, relative paths -> RAG_BASE_DIR
    # =====================================================================================
    @model_validator(mode="after")
    def _resolve_paths(self) -> "Settings":
        missing = []
        for env, attr in (("RAG_DATA_DIR", "data_dir"), ("RAG_DB_DIR", "index_dir")):
            if not (getattr(self, attr) or "").strip():
                missing.append(env)
        for section, env in SECTION_ENV.items():
            if getattr(self, f"dir_{section}") is None:
                missing.append(env)
        if missing:
            raise ValueError(
                "Faltan rutas en el .env (" + str(ENV_FILE) + "): " + ", ".join(missing)
                + ". Todas las rutas se declaran en el .env; ver .env.example (raíz del proyecto).")

        base = Path(self.base_dir).expanduser() if self.base_dir.strip() else ENV_FILE.parent
        if not base.is_absolute():
            base = ENV_FILE.parent / base
        self.base_dir = str(base.resolve())

        def res(v: str) -> str:
            p = Path(str(v).strip().strip('"').strip("'")).expanduser()
            return str((p if p.is_absolute() else Path(self.base_dir) / p).resolve())

        for attr in ("data_dir", "index_dir"):
            setattr(self, attr, res(getattr(self, attr)))
        self.datasheet_logo = (res(self.datasheet_logo) if self.datasheet_logo.strip()
                               else str(PROJECT_ROOT / "assets" / "logo_relant.png"))
        for attr in ("chroma_dir_env", "web_cache_dir_env", "datasheet_out_dir_env", "log_dir",
                     "datasheet_pdf_config", "semantic_cache_file_env", "trace_log_file_env"):
            if getattr(self, attr).strip():
                setattr(self, attr, res(getattr(self, attr)))

        data = Path(self.data_dir)
        for section in SECTION_ENV:
            dirs = []
            for d in _dir_list(getattr(self, f"dir_{section}")):
                p = Path(d)
                if p.is_absolute() or (len(d) > 1 and d[1] == ":"):
                    try:
                        d = p.resolve().relative_to(data).as_posix()
                    except ValueError:
                        raise ValueError(
                            f"{SECTION_ENV[section]}={d} está fuera de RAG_DATA_DIR ({data}). "
                            "Las carpetas de sección van relativas a RAG_DATA_DIR.") from None
                if ".." in d.split("/"):
                    raise ValueError(f"{SECTION_ENV[section]}: '..' no está permitido ({d})")
                dirs.append(d)
            setattr(self, f"dir_{section}", ",".join(dirs))
        excl = []
        for d in _dir_list(self.exclude_dirs):
            p = Path(d)
            if p.is_absolute() or (len(d) > 1 and d[1] == ":"):
                try:
                    d = p.resolve().relative_to(data).as_posix()
                except ValueError:
                    raise ValueError(f"RAG_EXCLUDE_DIRS={d} está fuera de RAG_DATA_DIR ({data})") \
                        from None
            excl.append(d)
        self.exclude_dirs = ",".join(excl)
        self.source_routing = (self.source_routing or "prefer").strip().lower()
        if self.source_routing not in ("prefer", "strict", "off"):
            raise ValueError("SOURCE_ROUTING debe ser prefer, strict u off")
        self.semantic_cache_invalidation = (self.semantic_cache_invalidation or "sources").strip().lower()
        if self.semantic_cache_invalidation not in ("sources", "index", "off"):
            raise ValueError("SEMANTIC_CACHE_INVALIDATION debe ser sources, index u off")
        return self

    # ---- sections ----
    def section_dirs(self, section: str) -> tuple[str, ...]:
        return _dir_list(getattr(self, f"dir_{section}", "") or "")

    def _section_spec(self) -> tuple:
        return tuple((s, self.section_dirs(s)) for s in SECTION_ENV)

    def section_of(self, rel: str) -> str:
        """Section of a file from its path relative to the data folder of its base
        ('@slug/' namespace allowed). Only the folders declared in the .env count."""
        r = str(rel or "").replace("\\", "/")
        if r.startswith("@") and "/" in r:
            r = r.split("/", 1)[1]
        folders = tuple(s.casefold() for s in r.split("/")[:-1] if s)
        for segs, section in _section_prefixes(self._section_spec()):
            if folders[:len(segs)] == segs:
                return section
        return OTHER_SECTION

    def is_excluded(self, rel: str) -> bool:
        """True if the file (path relative to RAG_DATA_DIR) is inside RAG_EXCLUDE_DIRS."""
        folders = tuple(s.casefold() for s in str(rel or "").replace("\\", "/").split("/")[:-1] if s)
        for d in _dir_list(self.exclude_dirs):
            segs = tuple(s.casefold() for s in d.split("/") if s and s != ".")
            if segs and folders[:len(segs)] == segs:
                return True
        return False

    def section_root(self, rel: str) -> tuple[str, str]:
        """(section, path inside the section folder) — e.g. manuales/abnox/x.pdf ->
        ('manuales', 'abnox/x.pdf')."""
        r = str(rel or "").replace("\\", "/")
        if r.startswith("@") and "/" in r:
            r = r.split("/", 1)[1]
        parts = [s for s in r.split("/") if s]
        folders = tuple(s.casefold() for s in parts[:-1])
        for segs, section in _section_prefixes(self._section_spec()):
            if folders[:len(segs)] == segs:
                return section, "/".join(parts[len(segs):])
        return OTHER_SECTION, "/".join(parts)

    def first_dir(self, section: str) -> str:
        d = self.section_dirs(section)
        return d[0] if d else ""

    # ---- derived paths (layout the deployed index already uses) ----
    @property
    def chroma_dir(self) -> str:
        return self.chroma_dir_env or str(Path(self.index_dir) / "chroma")

    @property
    def manifest_file(self) -> str:
        return str(Path(self.index_dir) / "manifest.json")

    @property
    def state_file(self) -> str:
        return str(Path(self.index_dir) / "state.json")

    @property
    def sqlite_cache_file(self) -> str:
        """Embedding / drawing-description cache (SQLite file next to the manifest)."""
        return self.manifest_file + ".cache.sqlite"

    @property
    def web_cache_dir(self) -> Path:
        return Path(self.web_cache_dir_env or Path(self.index_dir) / "web_cache")

    @property
    def datasheet_out_dir(self) -> Path:
        """Batch-generated HTML sheets. Outside data_dir on purpose: they must not be indexed
        (they would duplicate the CSV rows in every search)."""
        return Path(self.datasheet_out_dir_env or Path(self.index_dir) / "hojas_generadas")

    @property
    def semantic_cache_path(self) -> str:
        """Answer cache (SQLite). Separate from the embedding cache on purpose: it can be
        deleted at any time (cache-clear) without losing the paid embeddings."""
        return self.semantic_cache_file_env or str(Path(self.index_dir) / "semantic_cache.sqlite")

    @property
    def trace_log_path(self) -> str:
        if self.trace_log_file_env:
            return self.trace_log_file_env
        return str(Path(self.log_dir or (PROJECT_ROOT / "logs")) / "relant_rag_agent.log")

    @property
    def effective_validator_model(self) -> str:
        return self.semantic_cache_validator_model or self.llm_model

    @property
    def effective_agent_model(self) -> str:
        return self.agent_model or self.llm_model

    @property
    def effective_vision_model(self) -> str:
        return self.vision_model or self.llm_model

    def paths_report(self) -> dict:
        """Todas las rutas efectivas (comando `paths`)."""
        out = {"env_file": str(ENV_FILE), "env_file_exists": ENV_FILE.is_file(),
               "RAG_BASE_DIR": self.base_dir, "RAG_DATA_DIR": self.data_dir,
               "RAG_DB_DIR": self.index_dir, "RAG_DATASHEET_LOGO": self.datasheet_logo,
               "RAG_CHROMA_DIR": self.chroma_dir, "RAG_WEB_CACHE_DIR": str(self.web_cache_dir),
               "RAG_DATASHEET_OUT_DIR": str(self.datasheet_out_dir),
               "RAG_DATASHEET_PDF_CONFIG": self.datasheet_pdf_config or "(plantilla por defecto)",
               "RAG_LOG_DIR": self.log_dir or "(consola)",
               "EMBED_CACHE": self.sqlite_cache_file,
               "RAG_SEMANTIC_CACHE_FILE": (self.semantic_cache_path if self.semantic_cache
                                           else "(desactivada: SEMANTIC_CACHE=false)"),
               "RAG_TRACE_LOG": self.trace_log_path if self.trace_log else "(desactivado)",
               "RAG_EXCLUDE_DIRS": [str(Path(self.data_dir) / d) for d in _dir_list(self.exclude_dirs)]
               or "(ninguna)"}
        for section, env in SECTION_ENV.items():
            out[env] = [str(Path(self.data_dir) / d) for d in self.section_dirs(section)] \
                or "(no usada)"
        return out


def load_settings() -> Settings:
    try:
        return Settings()
    except Exception as e:                        # pydantic ValidationError -> clear message
        msg = str(e)
        if "Faltan rutas" in msg or "RAG_" in msg or "SOURCE_ROUTING" in msg:
            first = next((ln.strip() for ln in msg.splitlines()
                          if "Faltan rutas" in ln or "RAG_" in ln or "SOURCE_ROUTING" in ln), msg)
            raise ConfigError(first.replace("Value error, ", "")) from None
        raise


settings = load_settings()
