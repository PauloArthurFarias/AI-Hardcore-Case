"""
Módulo principal (Entry Point) da API REST do Sistema Multi-Agentes.
Responsável por expor a interface HTTP, validar o contrato de dados de entrada
(aplicando na prática o Spec-Driven Development) e delegar a requisição 
para a camada de orquestração de IA.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from agents import router_agent
import os

# 1. INICIALIZAÇÃO DA APLICAÇÃO
# Instancia o servidor FastAPI. O parâmetro 'title' é utilizado para gerar
# automaticamente a documentação interativa (Swagger UI/OpenAPI) acessível na rota /docs.
app = FastAPI(title="Getnet AI Agent System")

# 2. CONTRATO DE DADOS (SPEC-DRIVEN DEVELOPMENT - SDD)
# Defini estritamente o formato esperado do payload da requisição.
# O Pydantic valida em tempo de execução se o JSON recebido contém exatamente
# os campos 'message' e 'user_id' com os tipos corretos (string). 
# Requisições fora desse padrão são rejeitadas automaticamente na porta (HTTP 422),
# economizando processamento e blindando o LLM contra inputs inesperados ou maliciosos.
class QueryRequest(BaseModel):
    message: str
    user_id: str

# 3. ENDPOINT DE COMUNICAÇÃO
# Expõe a rota POST /chat. A função é declarada como 'async' para não bloquear 
# a thread principal do servidor (event loop). Isso permite que o FastAPI atenda 
# novos clientes enquanto aguarda o tempo de resposta (I/O bound) da API do Groq/Tavily.
@app.post("/chat")
async def chat_endpoint(req: QueryRequest):
    try:
        # Validação de Segurança (Fail-Fast)
        # Verifica a integridade do ambiente antes de gastar recursos de roteamento.
        # Evita crashes obscuros no meio da execução do LangChain por falta de credenciais.
        if not os.getenv("GROQ_API_KEY"):
            return {"error": "GROQ_API_KEY não configurada no ambiente."}
            
        # Delegação (Hand-off)
        # Repassa os dados validados e higienizados para o cérebro do sistema (Router Agent).
        # A partir desta linha, a lógica de Inteligência Artificial assume o controle.
        response = router_agent(req.message, req.user_id)
        
        # O FastAPI converte automaticamente este dicionário para uma resposta HTTP JSON (application/json)
        return {"response": response}
        
    except Exception as e:
        # Blindagem de Exceções (Graceful Degradation)
        # Captura qualquer erro não tratado durante a execução da IA (ex: timeout de rede, limite de tokens)
        # e o encapsula em um JSON amigável, impedindo que o processo do servidor sofra um crash (fatal error).
        return {"error": str(e)}