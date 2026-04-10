import os
import json
from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.services.drive import listar_arquivos, baixar_arquivo
from api.services.gemini import get_gemini_client, criar_store, apagar_store, indexar_arquivo
from google.genai import types

router = APIRouter()

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


# ── Helpers do JSON local ────────────────────────────────────────────────────

def carregar_stores() -> dict:
    if not os.path.exists(STORES_FILE):
        return {}
    with open(STORES_FILE, "r") as f:
        return json.load(f)


def salvar_stores(data: dict):
    with open(STORES_FILE, "w") as f:
        json.dump(data, f, indent=2)


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

        for arquivo in arquivos:
            caminho = baixar_arquivo(arquivo["id"], arquivo["name"])
            caminhos_temp.append(caminho)
            indexar_arquivo(gemini, gemini_store_name, caminho, arquivo["name"])
            arquivos_indexados += 1

        # ── Limpa arquivos temporários ───────────────────────────────────────
        for caminho in caminhos_temp:
            if os.path.exists(caminho):
                os.remove(caminho)

        # ── Salva no JSON: folder_id → { gemini_store_name, store_display_name } ──
        stores[body.drive_folder_id] = {
            "gemini_store_name": gemini_store_name,
            "store_display_name": body.store_name
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
        raise HTTPException(status_code=500, detail=str(e))


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
            "store_display_name": dados["store_display_name"]
        })

    return {"status": "success", "stores": resultado}


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
        contents = f"""Você é um assistente de estudos acadêmico.

REGRAS OBRIGATÓRIAS:
- Responda SEMPRE em português brasileiro, sem exceção
- Use APENAS as informações dos materiais indexados da disciplina
- Se não encontrar a informação, diga: "Não encontrei essa informação nos materiais da disciplina."
- Nunca use conhecimento próprio para complementar respostas
- Nunca peça para o aluno fornecer materiais
- Você PODE reformatar, resumir ou reorganizar informações que já estão nos materiais
{historico_texto}
Pergunta atual do aluno: {body.prompt}"""

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

        return ChatResponse(
            status="success",
            resposta=response.text
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))