import os
import io
import re
import unicodedata
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

CREDENTIALS_FILE = os.getenv("CREDENTIALS_FILE", "credentials.json")

def get_drive_client():
    creds = service_account.Credentials.from_service_account_file(
        CREDENTIALS_FILE,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=creds)


def listar_arquivos(folder_id: str) -> list[dict]:
    """Retorna lista de arquivos de uma pasta do Drive."""
    drive = get_drive_client()
    resultado = drive.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id, name, mimeType, webViewLink, webContentLink)"
    ).execute()
    return resultado.get("files", [])


def gerar_link_visualizacao(file_id: str) -> str:
    """Gera um link padrão de visualização no Google Drive."""
    if not file_id:
        return ""
    return f"https://drive.google.com/file/d/{file_id}/view"


def baixar_arquivo(file_id: str, file_name: str) -> str:
    """Baixa um arquivo do Drive e retorna o caminho local."""
    drive = get_drive_client()
    os.makedirs("temp_files", exist_ok=True)

    def _sanitizar_nome_arquivo(nome: str) -> str:
        nome = (nome or "").strip()
        # separa extensão
        base, ext = os.path.splitext(nome)
        # remove acentos e força ASCII
        base = unicodedata.normalize("NFKD", base)
        base = "".join(ch for ch in base if not unicodedata.combining(ch))
        base = base.encode("ascii", errors="ignore").decode("ascii")
        base = base.lower()
        # troca espaços por underscore e remove caracteres problemáticos
        base = re.sub(r"\s+", "_", base)
        base = re.sub(r"[^a-z0-9._-]", "", base)
        base = base.strip("._-")
        if not base:
            base = "arquivo"

        ext = (ext or "").lower()
        ext = re.sub(r"[^a-z0-9.]", "", ext)
        if ext and not ext.startswith("."):
            ext = "." + ext
        return base + ext

    safe_name = _sanitizar_nome_arquivo(file_name)
    # garante unicidade e evita colisões
    prefix = (file_id or "").strip()
    if prefix:
        safe_name = f"{prefix}_{safe_name}"

    caminho = os.path.join("temp_files", safe_name)
    request = drive.files().get_media(fileId=file_id)
    with io.FileIO(caminho, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return caminho