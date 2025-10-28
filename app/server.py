from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json

import logging
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from src.kg_gen.kg_gen import KGGen
from src.kg_gen.models import Graph
from src.kg_gen.utils.visualize_kg import _build_view_model

APP_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = (
    APP_DIR.parent / "src" / "kg_gen" / "utils" / "template.html"
).resolve()
DATA_ROOT = (APP_DIR.parent / "app" / "examples").resolve()


@dataclass(frozen=True)
class ExampleGraph:
    slug: str
    title: str
    path: Path
    wiki_url: str


# TODO: this will be read from huggingface once it is uploaded.
EXAMPLE_GRAPHS: tuple[ExampleGraph, ...] = ()
for file in DATA_ROOT.glob("*.json"):
    EXAMPLE_GRAPHS += (
        ExampleGraph(
            slug=file.stem,
            title=file.stem,
            path=file,
            wiki_url=f"https://en.wikipedia.org/wiki/{file.stem}",
        ),
    )


EXAMPLE_INDEX = {
    example.slug: example for example in EXAMPLE_GRAPHS if example.path.exists()
}

if len(EXAMPLE_INDEX) < len(EXAMPLE_GRAPHS):
    missing = [
        example.slug for example in EXAMPLE_GRAPHS if example.slug not in EXAMPLE_INDEX
    ]
    logger = logging.getLogger("kg_gen_app")
    logger.warning("Example graphs missing on disk: %s", ", ".join(missing))

if not TEMPLATE_PATH.exists():
    raise RuntimeError(f"Template not found at {TEMPLATE_PATH}")

logger = logging.getLogger("kg_gen_app")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


app = FastAPI(title="kg-gen explorer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
kg_gen = KGGen()


@app.get("/", response_class=HTMLResponse)
async def serve_index() -> HTMLResponse:
    index_path = APP_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="index.html missing")
    logger.debug("Serving index page")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/template")
async def serve_template() -> FileResponse:
    logger.debug("Serving visualization template from %s", TEMPLATE_PATH)
    return FileResponse(TEMPLATE_PATH, media_type="text/html")


@app.get("/api/examples")
async def list_examples() -> JSONResponse:
    logger.debug("Listing built-in example graphs")
    items = [
        {"slug": example.slug, "title": example.title, "wiki_url": example.wiki_url}
        for example in sorted(
            EXAMPLE_INDEX.values(), key=lambda item: item.title.lower()
        )
    ]
    return JSONResponse(items)


@app.get("/api/examples/{slug}")
async def load_example(slug: str) -> JSONResponse:
    example = EXAMPLE_INDEX.get(slug)
    if example is None:
        raise HTTPException(status_code=404, detail=f"Example '{slug}' not found")

    if not example.path.exists():
        logger.error("Example graph missing: %s", example.path)
        raise HTTPException(status_code=404, detail=f"Example '{slug}' is unavailable")

    try:
        payload = json.loads(example.path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.exception("Failed to parse example graph %s", slug)
        raise HTTPException(
            status_code=500, detail=f"Example '{slug}' is invalid: {exc}"
        )

    logger.info("Loaded example graph '%s' from %s", slug, example.path)
    return JSONResponse(payload)


@app.post("/api/graph/view")
async def build_view(graph: Graph) -> JSONResponse:
    """Convert a raw KGGen graph payload into the template view model."""
    logger.info(
        "Received request to build view: entities=%s relations=%s",
        len(graph.entities),
        len(graph.relations),
    )
    try:
        view = _build_view_model(graph)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to build view model")
        raise HTTPException(status_code=500, detail=f"Failed to build view: {exc}")
    logger.info(
        "View model ready: nodes=%s edges=%s", len(view["nodes"]), len(view["edges"])
    )
    return JSONResponse({"view": view, "graph": graph.model_dump(mode="json")})


def _clean_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _parse_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.lower() in {"true", "1", "yes", "on"}


@app.post("/api/generate")
async def generate_graph(
    api_key: str = Form(..., description="OpenAI API key"),
    model: str = Form("openai/gpt-4o"),
    context: Optional[str] = Form(None),
    chunk_size: Optional[str] = Form(None),
    temperature: Optional[str] = Form(None),
    cluster: Optional[str] = Form(None),
    source_text: Optional[str] = Form(None),
    text_file: Optional[UploadFile] = File(None),
    retrieval_model: Optional[str] = Form("sentence-transformers/all-mpnet-base-v2"),
) -> JSONResponse:
    text_fragments: list[str] = []

    cleaned_text = _clean_str(source_text)
    if cleaned_text:
        text_fragments.append(cleaned_text)

    if text_file is not None:
        try:
            contents = await text_file.read()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to read uploaded text file")
            raise HTTPException(
                status_code=400, detail=f"Reading text file failed: {exc}"
            )
        try:
            decoded = contents.decode("utf-8")
        except UnicodeDecodeError as exc:
            logger.exception("Uploaded text file must be UTF-8")
            raise HTTPException(
                status_code=400, detail=f"Text file must be UTF-8: {exc}"
            )
        cleaned_file_text = _clean_str(decoded)
        if cleaned_file_text:
            text_fragments.append(cleaned_file_text)

    if not text_fragments:
        raise HTTPException(
            status_code=400, detail="Provide inline text or upload a .txt file"
        )

    request_text = "\n\n".join(text_fragments)

    numeric_chunk: Optional[int] = None
    if chunk_size:
        try:
            numeric_chunk = int(chunk_size)
        except ValueError as exc:
            logger.warning("Invalid chunk_size received: %s", chunk_size)
            raise HTTPException(
                status_code=400, detail=f"chunk_size must be an integer: {exc}"
            )
        if numeric_chunk <= 0:
            numeric_chunk = None

    numeric_temperature: Optional[float] = None
    if temperature:
        try:
            numeric_temperature = float(temperature)
        except ValueError as exc:
            logger.warning("Invalid temperature received: %s", temperature)
            raise HTTPException(
                status_code=400, detail=f"temperature must be numeric: {exc}"
            )

    # Validate temperature for gpt-5 family models
    if "gpt-5" in model:
        if numeric_temperature is not None and numeric_temperature < 1.0:
            raise HTTPException(
                status_code=400,
                detail="Temperature must be 1.0 or higher for gpt-5 family models",
            )
        # Set default temperature to 1.0 for gpt-5 models if not specified
        if numeric_temperature is None:
            numeric_temperature = 1.0

    kg_gen.init_model(
        model=model,
        api_key=api_key,
        temperature=numeric_temperature,
        retrieval_model=retrieval_model,
    )

    logger.info(
        "Generating graph via KGGen: model=%s cluster=%s chunk_size=%s context_len=%s text_len=%s temperature=%s retrieval_model=%s",
        model,
        _parse_bool(cluster),
        numeric_chunk,
        len((_clean_str(context) or "")),
        len(request_text),
        numeric_temperature,
        retrieval_model,
    )
    try:
        graph = kg_gen.generate(
            input_data=request_text,
            model=model,
            api_key=api_key,
            context=_clean_str(context) or "",
            chunk_size=numeric_chunk,
            cluster=_parse_bool(cluster),
            temperature=numeric_temperature,
        )
    except ValidationError as exc:
        logger.exception("KGGen returned validation error")
        raise HTTPException(status_code=400, detail=f"Invalid graph result: {exc}")
    except Exception as exc:
        logger.exception("KGGen generation failed")
        raise HTTPException(status_code=500, detail=f"KGGen failed: {exc}")

    try:
        view = _build_view_model(graph)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to build view model after generation")
        raise HTTPException(status_code=500, detail=f"Failed to build view: {exc}")

    logger.info(
        "Graph generation complete: entities=%s relations=%s",
        len(graph.entities),
        len(graph.relations),
    )
    return JSONResponse({"view": view, "graph": graph.model_dump(mode="json")})


# Serve static files (CSS, JS, etc.) - must be mounted after all routes
app.mount("/", StaticFiles(directory=APP_DIR, html=True), name="static")
