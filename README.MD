# AcadêmicoAI — API

API do chatbot acadêmico construída com FastAPI + Gemini File Search + Google Drive API.

## Pré-requisitos

- Python 3.12+
- Conta Google com acesso ao [Google AI Studio](https://aistudio.google.com)
- Projeto no [Google Cloud Console](https://console.cloud.google.com) com a **Google Drive API** ativada

## Instalação

```bash
# Clone o repositório
git clone <url-do-repositorio>

# Crie e ative o ambiente virtual
python3 -m venv venv
source venv/bin/activate

# Instale as dependências
pip install fastapi uvicorn google-genai google-api-python-client google-auth python-dotenv tenacity
```

## Configuração

### 1. Credenciais do Google Drive

- Acesse o [Google Cloud Console](https://console.cloud.google.com)
- Selecione seu projeto
- Vá em **APIs e serviços → Credenciais**
- Clique na sua Service Account → aba **Chaves → Adicionar chave → JSON**
- Renomeie o arquivo baixado para `credentials.json` e coloque na raiz do projeto

### 2. Variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
GEMINI_API_KEY=sua-chave-aqui
CREDENTIALS_FILE=credentials.json
```

| Variável | Descrição |
|---|---|
| `GEMINI_API_KEY` | Chave da API do Gemini (obtenha em [aistudio.google.com](https://aistudio.google.com)) |
| `CREDENTIALS_FILE` | Caminho para o arquivo de credenciais do Google Drive |

### 3. Compartilhar pasta do Drive

- Abra a pasta do Drive que contém os materiais da disciplina
- Clique em **Compartilhar**
- Adicione o e-mail da Service Account (encontrado dentro do `credentials.json` no campo `client_email`)
- Permissão de **Visualizador** é suficiente

## Como rodar

```bash
# Com o venv ativo
uvicorn api.main:app --reload
```

A API estará disponível em `http://localhost:8000`.  
Documentação interativa em `http://localhost:8000/docs`.

## Endpoints

### `POST /store/indexar`
Cria ou recria uma store a partir de uma pasta do Google Drive.

**Body:**
```json
{
  "gemini_api_key": "sua-chave",
  "drive_folder_id": "id-da-pasta-no-drive",
  "store_name": "Nome da Disciplina"
}
```

**Como obter o `drive_folder_id`:** abra a pasta no Drive e copie o ID da URL:
```
https://drive.google.com/drive/folders/ESTE-E-O-ID
```

---

### `GET /store/listar`
Retorna todas as stores indexadas.

---

### `POST /store/chat`
Envia uma pergunta e recebe uma resposta baseada nos materiais da disciplina.

**Body:**
```json
{
  "gemini_api_key": "sua-chave",
  "gemini_store_name": "fileSearchStores/abc123",
  "prompt": "Qual a ementa da disciplina?",
  "historico": [
    { "role": "user", "content": "Olá" },
    { "role": "assistant", "content": "Olá! Como posso ajudar?" }
  ]
}
```

> O campo `historico` pode ser um array vazio `[]`.

## Estrutura do projeto

```
api/
├── main.py              # Entry point da API
├── routes/
│   └── store.py         # Rotas: /store/indexar, /store/listar, /store/chat
└── services/
    ├── drive.py         # Integração com Google Drive API
    └── gemini.py        # Integração com Gemini File Search API
```

## Arquivos gerados automaticamente

| Arquivo | Descrição |
|---|---|
| `stores.json` | Mapeamento entre `drive_folder_id` e `gemini_store_name` |
| `temp_files/` | Arquivos temporários durante a indexação (apagados automaticamente) |