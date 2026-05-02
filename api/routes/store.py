import os
import json
import re
import unicodedata
import logging
import sys
from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.services.drive import listar_arquivos, baixar_arquivo, gerar_link_visualizacao
from api.services.gemini import get_gemini_client, criar_store, apagar_store, indexar_arquivo
from google.genai import types

router = APIRouter()

logger = logging.getLogger(__name__)

STORES_FILE = "stores.json"


# ── Modelos ──────────────────────────────────────────────────────────────────

class SyncRequest(BaseModel):
    gemini_api_key: str
    drive_folder_id: str
    store_name: str


class SyncResponse(BaseModel):
    status: str
    store_name: str
    store_display_name: str
    arquivos_indexados: int
    mensagem: str


class DeleteStoreRequest(BaseModel):
    gemini_api_key: str
    drive_folder_id: str | None = None
    gemini_store_name: str | None = None


class DeleteStoreResponse(BaseModel):
    status: str
    gemini_store_name: str
    store_display_name: str | None = None
    drive_folder_id: str | None = None
    mensagem: str


# ── Helpers do JSON local ────────────────────────────────────────────────────

def carregar_stores() -> dict:
    if not os.path.exists(STORES_FILE):
        return {}
    with open(STORES_FILE, "r") as f:
        return json.load(f)


def salvar_stores(data: dict):
    with open(STORES_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _encontrar_store_por_gemini_store_name(gemini_store_name: str) -> tuple[str, dict] | tuple[None, None]:
    stores = carregar_stores()
    for drive_folder_id, dados in stores.items():
        if dados.get("gemini_store_name") == gemini_store_name:
            return drive_folder_id, dados
    return None, None


def _normalizar_nome_arquivo(nome: str) -> str:
    texto = (nome or "").strip().lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(ch for ch in texto if not unicodedata.combining(ch))
    texto = texto.replace("_", " ")
    texto = re.sub(r"\s+", " ", texto)
    return texto


def _remover_extensao(nome: str) -> str:
    return re.sub(r"\.[a-z0-9]{1,6}$", "", nome, flags=re.I)


def _mapa_links_por_nome(dados_store: dict) -> dict[str, str]:
    """Retorna um mapa nome->link a partir do que foi salvo em stores.json."""
    arquivos = dados_store.get("files") or []
    mapa: dict[str, str] = {}
    for arq in arquivos:
        nome = _normalizar_nome_arquivo(arq.get("name"))
        if not nome:
            continue
        link = arq.get("webViewLink") or arq.get("webContentLink")
        if not link and arq.get("id"):
            link = gerar_link_visualizacao(arq["id"])
        if link:
            mapa[nome] = link
            mapa[_remover_extensao(nome)] = link
    return mapa


def _buscar_link_por_token(mapa: dict[str, str], token: str) -> str | None:
    nome = _normalizar_nome_arquivo(token)
    if not nome:
        return None

    # Match exato (com e sem extensão)
    link = mapa.get(nome) or mapa.get(_remover_extensao(nome))
    if link:
        return link

    # Match por substring (tolerante a pequenas variações)
    for chave, lnk in mapa.items():
        if nome in chave or chave in nome:
            return lnk
    return None


def _substituir_referencias_por_links(texto: str, dados_store: dict) -> str:
    """Troca 'Referência: [arquivo.pdf]' por URLs de Drive quando possível.

    Observação: usa URL "crua" (sem colchetes/HTML) para facilitar o front a tornar clicável.
    """
    if not texto or not dados_store:
        return texto

    mapa = _mapa_links_por_nome(dados_store)
    if not mapa:
        return texto

    # Procura linhas de referência(s) e substitui apenas o conteúdo dentro de colchetes.
    # Exemplos aceitos:
    # - Referência: [Arquivo.pdf]
    # - Referências: [A.pdf] [B.pdf]
    # - Referência: Arquivo.pdf
    ref_line_re = re.compile(r"(?im)^\s*Refer\u00eancia(?:s)?\s*:\s*(.+?)\s*$")

    def replace_line(match: re.Match) -> str:
        original_tail = match.group(1).strip()

        # Se já tiver URL explícita, mantém.
        if re.search(r"https?://", original_tail, flags=re.I):
            return match.group(0)

        # Extrai tokens em colchetes; se não houver, considera o tail inteiro como 1 token.
        tokens = re.findall(r"\[([^\]]+)\]", original_tail)
        if not tokens:
            tokens = [original_tail]

        links: list[str] = []
        for token in tokens:
            link = _buscar_link_por_token(mapa, token)
            if link:
                links.append(link)

        if not links:
            return match.group(0)

        if len(links) == 1:
            return f"Referência: {links[0]}"

        # Um link por linha costuma ser mais fácil de "linkificar" no front.
        linhas = ["Referências:"]
        for i, lnk in enumerate(links, start=1):
            linhas.append(f"{i}) {lnk}")
        return "\n".join(linhas)

    return ref_line_re.sub(replace_line, texto)


def _extrair_referencias(texto: str) -> list[str]:
    """Extrai URLs de referência do texto.

    Suporta formatos como:
    - "Referência: https://..."
    - "Referências:\n1) https://...\n2) https://..."
    """
    if not texto:
        return []

    lines = texto.splitlines()
    urls: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Caso 1: "Referência: <URL>" (URL na mesma linha)
        if re.match(r"(?i)^\s*refer\u00eancia\s*:\s*", line):
            urls.extend(re.findall(r"https?://\S+", line))
            i += 1
            continue

        # Caso 2: bloco "Referências:" com URLs nas linhas seguintes
        if re.match(r"(?i)^\s*refer\u00eancias\s*:\s*$", line):
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                if not next_line:
                    break

                # Para de capturar se começou outra seção típica
                if re.match(r"(?i)^(pergunta|resposta|observa\u00e7\u00e3o|nota)\s*:\s*", next_line):
                    break

                urls.extend(re.findall(r"https?://\S+", next_line))
                i += 1
            continue

        # Caso 3: "Referências: ...<URL> ..." (tudo na mesma linha)
        if re.match(r"(?i)^\s*refer\u00eancias\s*:\s*", line):
            urls.extend(re.findall(r"https?://\S+", line))
            i += 1
            continue

        i += 1

    # Dedup preservando ordem
    seen: set[str] = set()
    unique: list[str] = []
    for u in urls:
        u = u.strip().rstrip(")].,;")
        if u and u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def _remover_bloco_referencias(texto: str) -> str:
    """Remove um bloco de referências no final do texto.

    Mantém a resposta limpa para exibição e deixa as referências apenas no campo
    estruturado `referencias`.
    """
    if not texto:
        return texto

    lines = texto.splitlines()

    def normalize_heading(line: str) -> str:
        s = (line or "").strip()
        # remove marcadores comuns de markdown
        s = re.sub(r"^[#>*\-\s]+", "", s)
        s = re.sub(r"[*_`]+", "", s)
        s = s.strip().lower()
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        s = re.sub(r"\s+", " ", s)
        return s

    def is_ref_marker(line: str) -> bool:
        s = normalize_heading(line)
        # "referencia:" / "referencias:" / "referencias" (título)
        if s.startswith("referencia:") or s.startswith("referencias:"):
            return True
        if s in {"referencia", "referencias"}:
            return True
        return False

    def is_ref_related_line(line: str) -> bool:
        if not (line or "").strip():
            return True
        if is_ref_marker(line):
            return True
        # qualquer URL na linha
        if re.search(r"https?://\S+", line):
            return True
        # linha numerada/lista (mesmo sem URL, costuma estar junto do bloco)
        if re.match(r"^\s*(\d+\)|\[\d+\]|\-\s+|\*\s+)", line):
            return True
        return False

    j = len(lines) - 1
    found_marker = False

    # Consome somente um bloco de referências no FINAL
    while j >= 0 and is_ref_related_line(lines[j]):
        if is_ref_marker(lines[j]):
            found_marker = True
        j -= 1

    if not found_marker:
        return texto

    trimmed = lines[: j + 1]
    while trimmed and not trimmed[-1].strip():
        trimmed.pop()
    return "\n".join(trimmed)


# ── Rotas ─────────────────────────────────────────────────────────────────────

@router.post("/indexar", response_model=SyncResponse)
def indexar(body: SyncRequest):
    """
    Recebe gemini_api_key, drive_folder_id e store_name.
    - Se já existe uma store para esse folder_id → apaga e recria.
    - Se não existe → cria do zero.
    Vincula o store_name (nome de exibição) à store criada.
    """
    try:
        gemini = get_gemini_client(body.gemini_api_key)
        stores = carregar_stores()

        # ── Verifica se já existe uma store para esse folder_id ──────────────
        entrada_existente = stores.get(body.drive_folder_id)
        if entrada_existente:
            try:
                apagar_store(gemini, entrada_existente["gemini_store_name"])
            except Exception:
                pass
            del stores[body.drive_folder_id]
            salvar_stores(stores)

        # ── Lista arquivos do Drive ──────────────────────────────────────────
        arquivos = listar_arquivos(body.drive_folder_id)
        if not arquivos:
            raise HTTPException(
                status_code=404,
                detail="Nenhum arquivo encontrado na pasta do Drive."
            )

        # ── Cria nova store ──────────────────────────────────────────────────
        gemini_store_name = criar_store(gemini, display_name=body.store_name)

        # ── Indexa cada arquivo ──────────────────────────────────────────────
        arquivos_indexados = 0
        caminhos_temp = []

        arquivos_salvos = []

        for arquivo in arquivos:
            caminho = baixar_arquivo(arquivo["id"], arquivo["name"])
            caminhos_temp.append(caminho)
            indexar_arquivo(gemini, gemini_store_name, caminho, arquivo["name"])
            arquivos_indexados += 1

            arquivos_salvos.append({
                "id": arquivo.get("id"),
                "name": arquivo.get("name"),
                "mimeType": arquivo.get("mimeType"),
                "webViewLink": arquivo.get("webViewLink") or (gerar_link_visualizacao(arquivo.get("id")) if arquivo.get("id") else None),
                "webContentLink": arquivo.get("webContentLink"),
            })

        # ── Limpa arquivos temporários ───────────────────────────────────────
        for caminho in caminhos_temp:
            if os.path.exists(caminho):
                os.remove(caminho)

        # ── Salva no JSON: folder_id → { gemini_store_name, store_display_name } ──
        stores[body.drive_folder_id] = {
            "gemini_store_name": gemini_store_name,
            "store_display_name": body.store_name,
            "files": arquivos_salvos,
        }
        salvar_stores(stores)

        return SyncResponse(
            status="success",
            store_name=gemini_store_name,
            store_display_name=body.store_name,
            arquivos_indexados=arquivos_indexados,
            mensagem=f"{arquivos_indexados} arquivo(s) indexado(s) com sucesso."
        )

    except HTTPException:
        raise
    except Exception as e:
        try:
            logger.exception("Erro no endpoint /store/indexar")
        except Exception:
            pass
        raise HTTPException(status_code=500, detail={"type": type(e).__name__, "message": str(e)})


@router.get("/listar")
def listar():
    """Retorna todas as stores salvas com seus nomes de exibição."""
    stores = carregar_stores()

    if not stores:
        return {"status": "success", "stores": []}

    resultado = []
    for folder_id, dados in stores.items():
        resultado.append({
            "drive_folder_id": folder_id,
            "gemini_store_name": dados["gemini_store_name"],
            "store_display_name": dados["store_display_name"],
            "arquivos": len(dados.get("files") or []),
        })

    return {"status": "success", "stores": resultado}


@router.delete("/apagar", response_model=DeleteStoreResponse)
def apagar(body: DeleteStoreRequest):
    """Apaga uma store no Gemini e remove o vínculo no stores.json.

    Você pode informar:
    - drive_folder_id (recomendado), ou
    - gemini_store_name
    """
    try:
        if not body.drive_folder_id and not body.gemini_store_name:
            raise HTTPException(
                status_code=400,
                detail="Informe drive_folder_id ou gemini_store_name."
            )

        stores = carregar_stores()
        drive_folder_id = None
        dados_store = None

        if body.drive_folder_id:
            drive_folder_id = body.drive_folder_id
            dados_store = stores.get(drive_folder_id)
        else:
            drive_folder_id, dados_store = _encontrar_store_por_gemini_store_name(body.gemini_store_name)

        if not drive_folder_id or not dados_store:
            raise HTTPException(status_code=404, detail="Store não encontrada no stores.json.")

        gemini = get_gemini_client(body.gemini_api_key)
        gemini_store_name = dados_store.get("gemini_store_name")
        if not gemini_store_name:
            raise HTTPException(status_code=400, detail="Entrada inválida no stores.json (gemini_store_name ausente).")

        apagar_store(gemini, gemini_store_name)

        # remove do JSON local
        stores.pop(drive_folder_id, None)
        salvar_stores(stores)

        return DeleteStoreResponse(
            status="success",
            gemini_store_name=gemini_store_name,
            store_display_name=dados_store.get("store_display_name"),
            drive_folder_id=drive_folder_id,
            mensagem="Store apagada com sucesso."
        )

    except HTTPException:
        raise
    except Exception as e:
        try:
            logger.exception("Erro no endpoint /store/apagar")
        except Exception:
            pass
        raise HTTPException(status_code=500, detail={"type": type(e).__name__, "message": str(e)})


# ── Chat ──────────────────────────────────────────────────────────────────────

class Mensagem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    gemini_api_key: str
    gemini_store_name: str
    prompt: str
    historico: list[Mensagem] = []


class ChatResponse(BaseModel):
    status: str
    resposta: str
    referencias: list[str] = []


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    """
    Recebe gemini_store_name, prompt e histórico de mensagens.
    Retorna a resposta do modelo com base nos materiais da store.
    O histórico pode ser vazio.
    """
    try:
        gemini = get_gemini_client(body.gemini_api_key)

        # ── Monta o histórico como texto de contexto ─────────────────────────
        historico_texto = ""
        if body.historico:
            historico_texto = "\n\nHistórico da conversa:\n"
            for msg in body.historico[-6:]:  # limita às últimas 6 mensagens
                papel = "Aluno" if msg.role == "user" else "Assistente"
                historico_texto += f"{papel}: {msg.content}\n"

        # ── Monta o conteúdo enviado ao modelo ───────────────────────────────
        contents = f"""Assistente acadêmico. Responda em português brasileiro.

        REGRAS:
        - Responda direto, sem saudações ou formalidades
        - Busque nos materiais indexados primeiro; se não encontrar, use sua base de conhecimento e informe
        - Pode reformatar/resumir materiais
        - Use markdown e emojis (tom profissional)
        - Ao usar material: adicione uma linha em branco e inclua ao final:
        - 1 fonte: "Referência: <URL>"
        - N fontes: "Referências:\n1) <URL>\n2) <URL>"

        {historico_texto}
        Pergunta: {body.prompt}"""

        # ── Chama o modelo ───────────────────────────────────────────────────
        response = gemini.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[body.gemini_store_name]
                    )
                )]
            )
        )
        try:
            print_usage_metadata(getattr(response, "usage_metadata", None))
        except Exception:
            pass
        # ── Troca referências por nome por links reais do Drive (se possível) ─
        drive_folder_id, dados_store = _encontrar_store_por_gemini_store_name(body.gemini_store_name)
        resposta_final = response.text or ""
        if drive_folder_id and dados_store:
            # Fallback para stores antigas: se não houver metadados de arquivos salvos,
            # tenta buscar os links no Drive e persistir em stores.json.
            if not (dados_store.get("files") or []):
                try:
                    arquivos_drive = listar_arquivos(drive_folder_id)
                    dados_store["files"] = [
                        {
                            "id": a.get("id"),
                            "name": a.get("name"),
                            "mimeType": a.get("mimeType"),
                            "webViewLink": a.get("webViewLink") or (gerar_link_visualizacao(a.get("id")) if a.get("id") else None),
                            "webContentLink": a.get("webContentLink"),
                        }
                        for a in arquivos_drive
                    ]
                    stores = carregar_stores()
                    if drive_folder_id in stores:
                        stores[drive_folder_id]["files"] = dados_store["files"]
                        salvar_stores(stores)
                except Exception:
                    pass

            resposta_final = _substituir_referencias_por_links(resposta_final, dados_store)

        refs = _extrair_referencias(resposta_final)
        resposta_final = (_remover_bloco_referencias(resposta_final) or "").strip()

        return ChatResponse(
            status="success",
            resposta=resposta_final,
            referencias=refs,
        )

    except HTTPException:
        raise
    except Exception as e:
        try:
            logger.exception("Erro no endpoint /store/chat")
        except Exception:
            pass
        raise HTTPException(status_code=500, detail={"type": type(e).__name__, "message": str(e)})

def print_usage_metadata(usage):
    if not usage:
        return

    def _safe_print(line: str):
        """Imprime sem quebrar em ambientes com stdout/stderr ASCII."""
        try:
            print(line)
        except UnicodeEncodeError:
            try:
                sys.stdout.buffer.write((str(line) + "\n").encode("utf-8", errors="replace"))
                sys.stdout.buffer.flush()
            except Exception:
                # Último recurso: não imprime
                pass

    rows = [
        ("Prompt",          usage.prompt_token_count),
        ("File search",     usage.tool_use_prompt_token_count),
        ("Raciocinio",      usage.thoughts_token_count),
        ("Resposta gerada", usage.candidates_token_count),
        ("Total",           usage.total_token_count),
    ]

    width_label = max(len(r[0]) for r in rows)
    width_value = max(len(str(r[1])) for r in rows)

    sep = f"+{'-' * (width_label + 2)}+{'-' * (width_value + 2)}+"
    _safe_print(sep)
    _safe_print(f"| {'Campo':<{width_label}} | {'Tokens':>{width_value}} |")
    _safe_print(sep)
    for label, value in rows[:-1]:
        _safe_print(f"| {label:<{width_label}} | {value if value else 0:>{width_value}} |")
    _safe_print(sep)
    _safe_print(f"| {'Total':<{width_label}} | {rows[-1][1] if rows[-1][1] else 0:>{width_value}} |")
    _safe_print(sep)