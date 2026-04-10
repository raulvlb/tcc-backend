import time
import os
from google import genai


def get_gemini_client(api_key: str):
    return genai.Client(api_key=api_key)


def criar_store(client, display_name: str) -> str:
    """Cria uma nova File Search Store e retorna o nome dela."""
    store = client.file_search_stores.create(config={
        "display_name": display_name
    })
    return store.name


def apagar_store(client, store_name: str):
    """Apaga uma store existente com todos os seus documentos."""
    client.file_search_stores.delete(
        name=store_name,
        config={"force": True}
    )


def indexar_arquivo(client, store_name: str, caminho: str, display_name: str):
    """Faz upload e indexa um arquivo na store."""
    operation = client.file_search_stores.upload_to_file_search_store(
        file=caminho,
        file_search_store_name=store_name,
        config={"display_name": display_name}
    )
    while not operation.done:
        time.sleep(5)
        operation = client.operations.get(operation)