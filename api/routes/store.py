import os
import json
import re
import time
import pickle
import unicodedata
import logging
import mimetypes
import sys
from typing import Literal

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from google.genai import types

from api.services.drive import listar_arquivos, baixar_arquivo, gerar_link_visualizacao
from api.services.gemini import get_gemini_client

router = APIRouter()
logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────

STORES_FILE = "stores.json"
INDEX_DIR = "indexes"          # pasta onde ficam os vetores (.pkl por store)
EMBEDDING_MODEL = "gemini-embedding-2-preview"
GENERATION_MODEL = "gemini-2.5-flash"
TOP_K = 3                      # quantos arquivos mais similares usar no contexto
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.25"))  # limiar p/ considerar que há material relevante

# Tipos suportados diretamente pelo embed_content via Part.from_bytes
MIME_SUPORTADOS = {
    "image/jpeg", "image/png", "image/gif", "image/webp",
    "image/heic", "image/heif",
    "video/mp4", "video/mpeg", "video/quicktime", "video/avi",
    "video/webm", "video/x-matroska",
    "audio/mp3", "audio/mpeg", "audio/wav", "audio/ogg",
    "audio/flac", "audio/aac", "audio/webm",
    "application/pdf",
    "text/plain", "text/html", "text/csv", "text/markdown",
}

# Tipos que precisam ser convertidos antes de indexar
MIME_PLANILHA = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "application/vnd.google-apps.spreadsheet",
    "text/csv",
}

os.makedirs(INDEX_DIR, exist_ok=True)


# ── Modelos Pydantic ──────────────────────────────────────────────────────────

class SyncRequest(BaseModel):
    gemini_api_key: str
    drive_folder_id: str
    store_name: str


class SyncResponse(BaseModel):
    status: str
    store_name: str
    arquivos_indexados: int
    arquivos_ignorados: int
    mensagem: str


class DeleteStoreRequest(BaseModel):
    gemini_api_key: str
    drive_folder_id: str | None = None
    store_name: str | None = None


class DeleteStoreResponse(BaseModel):
    status: str
    store_name: str
    drive_folder_id: str | None = None
    mensagem: str


class Mensagem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    gemini_api_key: str
    store_name: str
    prompt: str
    historico: list[Mensagem] = []


class ChatResponse(BaseModel):
    status: str
    resposta: str
    referencias: list[str] = []


# ── Helpers de persistência ───────────────────────────────────────────────────

def carregar_stores() -> dict:
    if not os.path.exists(STORES_FILE):
        return {}
    with open(STORES_FILE, "r") as f:
        return json.load(f)


def salvar_stores(data: dict):
    with open(STORES_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _index_path(store_name: str) -> str:
    safe = re.sub(r"[^\w\-]", "_", store_name)
    return os.path.join(INDEX_DIR, f"{safe}.pkl")


def carregar_index(store_name: str) -> list[dict]:
    """Carrega lista de { name, mime, drive_id, link, embedding } do disco."""
    path = _index_path(store_name)
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        return pickle.load(f)


def salvar_index(store_name: str, index: list[dict]):
    with open(_index_path(store_name), "wb") as f:
        pickle.dump(index, f)


def apagar_index(store_name: str):
    path = _index_path(store_name)
    if os.path.exists(path):
        os.remove(path)


def _encontrar_store_por_nome(store_name: str) -> tuple[str, dict] | tuple[None, None]:
    stores = carregar_stores()
    for folder_id, dados in stores.items():
        if dados.get("store_name") == store_name:
            return folder_id, dados
    return None, None


# ── Helpers de MIME / conversão ───────────────────────────────────────────────

def _detectar_mime(caminho: str, mime_drive: str | None = None) -> str:
    if mime_drive:
        return mime_drive
    mime, _ = mimetypes.guess_type(caminho)
    return mime or "application/octet-stream"


def _converter_planilha_para_csv(caminho: str) -> tuple[bytes, str]:
    """Converte xlsx/xls/csv para bytes CSV. Retorna (bytes, 'text/csv')."""
    import pandas as pd
    df = pd.read_excel(caminho) if not caminho.endswith(".csv") else pd.read_csv(caminho)
    return df.to_csv(index=False).encode("utf-8"), "text/csv"


# ── Helpers de texto / referências ───────────────────────────────────────────

def _normalizar_nome(nome: str) -> str:
    texto = (nome or "").strip().lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(ch for ch in texto if not unicodedata.combining(ch))
    texto = texto.replace("_", " ")
    return re.sub(r"\s+", " ", texto)


def _remover_extensao(nome: str) -> str:
    return re.sub(r"\.[a-z0-9]{1,6}$", "", nome, flags=re.I)


def _mapa_links(index: list[dict]) -> dict[str, str]:
    mapa: dict[str, str] = {}
    for item in index:
        nome = _normalizar_nome(item.get("name", ""))
        link = item.get("link", "")
        if nome and link:
            mapa[nome] = link
            mapa[_remover_extensao(nome)] = link
    return mapa


def _buscar_link(mapa: dict[str, str], token: str) -> str | None:
    nome = _normalizar_nome(token)
    link = mapa.get(nome) or mapa.get(_remover_extensao(nome))
    if link:
        return link
    for chave, lnk in mapa.items():
        if nome in chave or chave in nome:
            return lnk
    return None


def _substituir_referencias(texto: str, index: list[dict]) -> str:
    if not texto or not index:
        return texto
    mapa = _mapa_links(index)
    if not mapa:
        return texto

    ref_re = re.compile(r"(?im)^\s*Refer[eê]ncia(?:s)?\s*:\s*(.+?)\s*$")

    def replace_line(match: re.Match) -> str:
        tail = match.group(1).strip()
        if re.search(r"https?://", tail, flags=re.I):
            return match.group(0)
        tokens = re.findall(r"\[([^\]]+)\]", tail) or [tail]
        links = [_buscar_link(mapa, t) for t in tokens if _buscar_link(mapa, t)]
        if not links:
            return match.group(0)
        if len(links) == 1:
            return f"Referência: {links[0]}"
        return "Referências:\n" + "\n".join(f"{i}) {l}" for i, l in enumerate(links, 1))

    return ref_re.sub(replace_line, texto)


def _extrair_referencias(texto: str) -> list[str]:
    if not texto:
        return []
    lines = texto.splitlines()
    urls: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"(?i)^\s*refer[eê]ncia\s*:\s*", line):
            urls.extend(re.findall(r"https?://\S+", line))
            i += 1
            continue
        if re.match(r"(?i)^\s*refer[eê]ncias\s*:\s*$", line):
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                if not next_line:
                    break
                urls.extend(re.findall(r"https?://\S+", next_line))
                i += 1
            continue
        if re.match(r"(?i)^\s*refer[eê]ncias\s*:\s*", line):
            urls.extend(re.findall(r"https?://\S+", line))
            i += 1
            continue
        i += 1

    seen: set[str] = set()
    unique: list[str] = []
    for u in urls:
        u = u.strip().rstrip(")].,;")
        if u and u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def _remover_bloco_referencias(texto: str) -> str:
    if not texto:
        return texto
    lines = texto.splitlines()

    def normalize(line: str) -> str:
        s = re.sub(r"^[#>*\-\s]+", "", (line or "").strip())
        s = re.sub(r"[*_`]+", "", s).strip().lower()
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        return re.sub(r"\s+", " ", s)

    def is_ref_marker(line: str) -> bool:
        s = normalize(line)
        return (
            s.startswith("referencia:") or s.startswith("referencias:")
            or s in {"referencia", "referencias"}
        )

    def is_ref_line(line: str) -> bool:
        stripped = (line or "").strip()
        if not stripped:
            return True
        if is_ref_marker(line):
            return True
        if re.search(r"https?://\S+", line):
            return True
        if re.match(r"^\s*(\d+\)|\[\d+\]|\-\s+|\*\s+)", line):
            return True
        return False

    j = len(lines) - 1
    found = False
    while j >= 0 and is_ref_line(lines[j]):
        if is_ref_marker(lines[j]):
            found = True
        j -= 1

    if not found:
        return texto

    trimmed = lines[: j + 1]
    while trimmed and not trimmed[-1].strip():
        trimmed.pop()
    return "\n".join(trimmed)


def _limpar_texto_historico(texto: str) -> str:
    """Remove URLs e bloco de Referência(s) do histórico para evitar vazamento."""
    if not texto:
        return texto
    texto = _remover_bloco_referencias(texto)
    # Remove URLs para não induzir o modelo a repetir refs antigas
    texto = re.sub(r"https?://\S+", "", texto)
    # Normaliza múltiplos espaços/linhas
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


# ── Core: embedding ───────────────────────────────────────────────────────────

def _gerar_embedding(gemini, dados: bytes, mime: str) -> list[float]:
    """Gera embedding multimodal para qualquer tipo suportado."""
    result = gemini.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[types.Part.from_bytes(data=dados, mime_type=mime)],
    )
    return result.embeddings[0].values


def _gerar_embedding_texto(gemini, texto: str) -> list[float]:
    """Gera embedding para texto puro (usado na query do chat)."""
    result = gemini.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texto,
    )
    return result.embeddings[0].values


def _buscar_similares(
    query_emb: list[float],
    index: list[dict],
    top_k: int = TOP_K,
    min_score: float | None = None,
) -> list[dict]:
    """Retorna os top_k itens mais similares ao vetor de query.

    Se min_score for informado, filtra resultados com score < min_score.
    """
    if not index:
        return []
    embs = np.array([item["embedding"] for item in index], dtype=np.float32)
    q = np.array([query_emb], dtype=np.float32)
    scores = cosine_similarity(q, embs)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    resultados: list[dict] = []
    for i in top_indices:
        score = float(scores[i])
        if min_score is not None and score < min_score:
            continue
        item = dict(index[i])
        item["_score"] = score
        resultados.append(item)
    return resultados


# ── Rota: /indexar ────────────────────────────────────────────────────────────

@router.post("/indexar", response_model=SyncResponse)
def indexar(body: SyncRequest):
    """
    Lista arquivos do Drive, gera embeddings multimodais com Gemini Embedding 2
    e persiste o index em disco. Apaga e recria se já existir.
    """
    try:
        gemini = get_gemini_client(body.gemini_api_key)
        stores = carregar_stores()

        # Apaga store existente para essa pasta
        entrada = stores.get(body.drive_folder_id)
        if entrada:
            apagar_index(entrada.get("store_name", ""))
            del stores[body.drive_folder_id]
            salvar_stores(stores)

        # Lista arquivos do Drive
        arquivos = listar_arquivos(body.drive_folder_id)
        if not arquivos:
            raise HTTPException(
                status_code=404,
                detail="Nenhum arquivo encontrado na pasta do Drive."
            )

        index: list[dict] = []
        arquivos_ignorados = 0

        for arquivo in arquivos:
            nome = arquivo["name"]
            mime_drive = arquivo.get("mimeType", "")
            caminho = None

            try:
                caminho = baixar_arquivo(arquivo["id"], nome)
                mime = _detectar_mime(caminho, mime_drive)

                # Planilhas → converte para CSV
                if mime in MIME_PLANILHA:
                    dados, mime = _converter_planilha_para_csv(caminho)
                elif mime in MIME_SUPORTADOS:
                    with open(caminho, "rb") as f:
                        dados = f.read()
                else:
                    logger.warning(f"Tipo não suportado, ignorando: {nome} ({mime})")
                    arquivos_ignorados += 1
                    continue

                # Gera embedding multimodal — uma única chamada para todos os tipos
                embedding = _gerar_embedding(gemini, dados, mime)

                index.append({
                    "name": nome,
                    "mime": mime,
                    "drive_id": arquivo["id"],
                    "link": (
                        arquivo.get("webViewLink")
                        or gerar_link_visualizacao(arquivo["id"])
                    ),
                    "embedding": embedding,
                })

                logger.info(f"Indexado: {nome} ({mime})")

            except Exception as e:
                logger.warning(f"Erro ao indexar '{nome}': {e}")
                arquivos_ignorados += 1
            finally:
                if caminho and os.path.exists(caminho):
                    os.remove(caminho)

        if not index:
            raise HTTPException(
                status_code=422,
                detail="Nenhum arquivo pôde ser indexado. Verifique os tipos suportados."
            )

        # Persiste index e metadados
        salvar_index(body.store_name, index)

        stores[body.drive_folder_id] = {
            "store_name": body.store_name,
            "files": [
                {
                    "name": item["name"],
                    "mime": item["mime"],
                    "drive_id": item["drive_id"],
                    "link": item["link"],
                }
                for item in index
            ],
        }
        salvar_stores(stores)

        return SyncResponse(
            status="success",
            store_name=body.store_name,
            arquivos_indexados=len(index),
            arquivos_ignorados=arquivos_ignorados,
            mensagem=f"{len(index)} arquivo(s) indexado(s) com sucesso."
            + (f" {arquivos_ignorados} ignorado(s)." if arquivos_ignorados else ""),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erro no endpoint /store/indexar")
        raise HTTPException(status_code=500, detail={"type": type(e).__name__, "message": str(e)})


# ── Rota: /listar ─────────────────────────────────────────────────────────────

@router.get("/listar")
def listar():
    """Retorna todas as stores salvas."""
    stores = carregar_stores()
    if not stores:
        return {"status": "success", "stores": []}

    resultado = []
    for folder_id, dados in stores.items():
        resultado.append({
            "drive_folder_id": folder_id,
            "store_name": dados.get("store_name"),
            "arquivos": len(dados.get("files") or []),
        })

    return {"status": "success", "stores": resultado}


# ── Rota: /apagar ─────────────────────────────────────────────────────────────

@router.delete("/apagar", response_model=DeleteStoreResponse)
def apagar(body: DeleteStoreRequest):
    """Apaga index do disco e remove vínculo no stores.json."""
    try:
        if not body.drive_folder_id and not body.store_name:
            raise HTTPException(
                status_code=400,
                detail="Informe drive_folder_id ou store_name."
            )

        stores = carregar_stores()
        drive_folder_id = None
        dados_store = None

        if body.drive_folder_id:
            drive_folder_id = body.drive_folder_id
            dados_store = stores.get(drive_folder_id)
        else:
            drive_folder_id, dados_store = _encontrar_store_por_nome(body.store_name)

        if not drive_folder_id or not dados_store:
            raise HTTPException(status_code=404, detail="Store não encontrada.")

        store_name = dados_store.get("store_name", "")
        apagar_index(store_name)

        stores.pop(drive_folder_id, None)
        salvar_stores(stores)

        return DeleteStoreResponse(
            status="success",
            store_name=store_name,
            drive_folder_id=drive_folder_id,
            mensagem="Store apagada com sucesso.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erro no endpoint /store/apagar")
        raise HTTPException(status_code=500, detail={"type": type(e).__name__, "message": str(e)})


# ── Rota: /chat ───────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    """
    1. Embeda o prompt com Embedding 2
    2. Busca os TOP_K arquivos mais similares no index
    3. Re-baixa esses arquivos do Drive
    4. Passa os bytes diretamente ao Gemini 2.5 Flash para geração
    O modelo "vê" o conteúdo real (foto, vídeo, áudio, PDF) ao responder.
    """
    try:
        gemini = get_gemini_client(body.gemini_api_key)

        # Carrega index da store
        index = carregar_index(body.store_name)
        if not index:
            raise HTTPException(
                status_code=404,
                detail=f"Store '{body.store_name}' não encontrada ou vazia."
            )

        # ── 1. Embedding da query ────────────────────────────────────────────
        query_emb = _gerar_embedding_texto(gemini, body.prompt)

        # ── 2. Busca semântica ───────────────────────────────────────────────
        top_itens = _buscar_similares(query_emb, index, top_k=TOP_K, min_score=MIN_SIMILARITY)

        # ── 3. Re-baixa arquivos relevantes e monta partes de contexto ───────
        partes_contexto: list = []
        referencias: list[str] = []
        caminhos_temp: list[str] = []

        for item in top_itens:
            try:
                caminho = baixar_arquivo(item["drive_id"], item["name"])
                caminhos_temp.append(caminho)

                mime = item["mime"]

                # Planilha foi indexada como CSV — re-converte para passar inline
                if mime == "text/csv" and not item["name"].endswith(".csv"):
                    dados, mime = _converter_planilha_para_csv(caminho)
                else:
                    with open(caminho, "rb") as f:
                        dados = f.read()

                partes_contexto.append(
                    types.Part.from_bytes(data=dados, mime_type=mime)
                )
                if item.get("link"):
                    referencias.append(item["link"])

            except Exception as e:
                logger.warning(f"Não foi possível carregar '{item['name']}' para o contexto: {e}")

        # ── 4. Monta histórico ───────────────────────────────────────────────
        historico_texto = ""
        if body.historico:
            historico_texto = "\n\nHistórico da conversa:\n"
            for msg in body.historico[-6:]:
                papel = "Aluno" if msg.role == "user" else "Assistente"
                conteudo = _limpar_texto_historico(msg.content)
                if conteudo:
                    historico_texto += f"{papel}: {conteudo}\n"

        # ── 5. Monta prompt final ────────────────────────────────────────────
        tem_materiais = len(partes_contexto) > 0
        if tem_materiais:
            nomes_contexto = ", ".join(i["name"] for i in top_itens)
            regra_contexto = f"- Os materiais acima são os mais relevantes para a pergunta ({nomes_contexto})"
            regra_refs = (
                "- Ao usar material: adicione uma linha em branco e inclua ao final:\n"
                "  - 1 fonte: \"Referência: <URL do Drive>\"\n"
                "  - N fontes: \"Referências:\\n1) <URL>\\n2) <URL>\""
            )
        else:
            regra_contexto = "- Não foram encontrados materiais relevantes nas suas stores para esta pergunta"
            regra_refs = "- Não inclua seção de Referência(s) e não liste URLs de fontes"

        system_prompt = f"""Assistente acadêmico. Responda em português brasileiro.

REGRAS:
- Responda direto, sem saudações ou formalidades
{regra_contexto}
- Se não houver material suficiente, use seu conhecimento e informe que não usou a store
- Pode reformatar/resumir materiais quando existirem
- Use markdown e emojis (tom profissional)
{regra_refs}
{historico_texto}
Pergunta: {body.prompt}"""

        # ── 6. Gera resposta com os arquivos reais no contexto ───────────────
        contents = [
            *partes_contexto,
            types.Part(text=system_prompt),
        ]

        response = gemini.models.generate_content(
            model=GENERATION_MODEL,
            contents=contents,
        )

        _print_usage(getattr(response, "usage_metadata", None))

        # ── 7. Pós-processamento de referências ──────────────────────────────
        resposta_final = response.text or ""
        refs: list[str] = []
        if tem_materiais:
            resposta_final = _substituir_referencias(resposta_final, index)
            # Só retorna referências se o modelo realmente citou fontes na resposta
            extraidas = _extrair_referencias(resposta_final)
            # Filtra para apenas URLs dos materiais realmente carregados nesta resposta
            permitidas = set(referencias)
            refs = [u for u in extraidas if u in permitidas]
        # Sempre remove bloco de referências do texto final
        resposta_final = (_remover_bloco_referencias(resposta_final) or "").strip()
        print(refs)
        return ChatResponse(
            status="success",
            resposta=resposta_final,
            referencias=refs,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erro no endpoint /store/chat")
        raise HTTPException(status_code=500, detail={"type": type(e).__name__, "message": str(e)})
    finally:
        for caminho in caminhos_temp:
            if os.path.exists(caminho):
                os.remove(caminho)


# ── Utilitário: print de tokens ───────────────────────────────────────────────

def _print_usage(usage):
    if not usage:
        return

    def _safe(line: str):
        try:
            print(line)
        except UnicodeEncodeError:
            try:
                sys.stdout.buffer.write((line + "\n").encode("utf-8", errors="replace"))
                sys.stdout.buffer.flush()
            except Exception:
                pass

    rows = [
        ("Prompt",          getattr(usage, "prompt_token_count", 0)),
        ("Raciocinio",      getattr(usage, "thoughts_token_count", 0)),
        ("Resposta gerada", getattr(usage, "candidates_token_count", 0)),
        ("Total",           getattr(usage, "total_token_count", 0)),
    ]
    w_l = max(len(r[0]) for r in rows)
    w_v = max(len(str(r[1] or 0)) for r in rows)
    sep = f"+{'-' * (w_l + 2)}+{'-' * (w_v + 2)}+"
    _safe(sep)
    _safe(f"| {'Campo':<{w_l}} | {'Tokens':>{w_v}} |")
    _safe(sep)
    for label, value in rows[:-1]:
        _safe(f"| {label:<{w_l}} | {value or 0:>{w_v}} |")
    _safe(sep)
    _safe(f"| {'Total':<{w_l}} | {rows[-1][1] or 0:>{w_v}} |")
    _safe(sep)