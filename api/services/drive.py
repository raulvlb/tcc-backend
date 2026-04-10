import os
import io
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
        fields="files(id, name, mimeType)"
    ).execute()
    return resultado.get("files", [])


def baixar_arquivo(file_id: str, file_name: str) -> str:
    """Baixa um arquivo do Drive e retorna o caminho local."""
    drive = get_drive_client()
    os.makedirs("temp_files", exist_ok=True)
    caminho = f"temp_files/{file_name}"
    request = drive.files().get_media(fileId=file_id)
    with io.FileIO(caminho, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return caminho