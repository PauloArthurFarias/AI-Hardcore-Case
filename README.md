# Getnet Multi-Agent Support System

Este repositório contém a solução do case técnico "AI Hardcore Engineer", apresentando uma arquitetura orquestrada de múltiplos agentes projetada para triagem semântica, suporte ao cliente e recuperação de conhecimento institucional (RAG) da Getnet.

## 1. Como Construir e Executar

A aplicação foi totalmente desenhada sob os princípios do *Spec-Driven Development (SDD)* e containerizada para garantir reprodutibilidade isolada do ambiente.

**Pré-requisitos:** Docker e Docker Compose instalados.

1. Clone o repositório.
2. Crie um arquivo `.env` na raiz do projeto baseado no `.env.example`:
   ```bash
   GROQ_API_KEY="sua_chave_groq"
   TAVILY_API_KEY="sua_chave_tavily"
Construa e suba a infraestrutura:

Comando no terminal:    

docker compose up --build    

A API estará exposta em http://localhost:8000.

Especificação (SDD): Acesse a documentação interativa (Swagger/OpenAPI) gerada automaticamente pelo FastAPI em http://localhost:8000/docs para inspecionar os contratos estritos da API.

## 2. Como Utilizar (Message Workflow)
Envie requisições via POST para o endpoint /chat respeitando o payload exigido (validado via Pydantic):

Comando no terminal:
  
curl -X POST http://localhost:8000/chat \
-H "Content-Type: application/json" \
-d '{"message": "What is the difference between the Get Clássica and the Get Smart?", "user_id": "cliente1988"}'

Workflow Interno de Mensageria:  
Endpoint FastAPI (SDD) -> Router Agent -> Agente Especialista (Knowledge OU Support) -> Execução de Tools -> Personality Node -> Resposta Final.

## 3. Arquitetura de Orquestração e Decisões de Design
A arquitetura foi concebida refletindo fielmente o diagrama de workflow de agentes exigido no case. As responsabilidades foram divididas para maximizar velocidade, reduzir consumo de tokens e garantir a padronização da marca, evitando os gargalos de um agente monolítico.

Agent 1 — Router Agent (Entry Point): Atua exclusivamente como um classificador semântico via prompt engineering (Zero-Shot). Ele lê a intenção do usuário e faz o roteamento determinístico para o agente especialista apropriado.

Agent 2 — Knowledge Agent: Equipado com duas ferramentas (Tools). Lida com dúvidas gerais usando a search_tool (integrada à Tavily API para buscas web resilientes contra bloqueios) e RAG corporativo para produtos institucionais.

Agent 3 — Customer Support Agent: Especialista em atendimento técnico (ReAct). Possui duas custom tools (check_machine_status e check_sales_balance) que extraem o user_id do contexto e consultam dados de um mock (simulando um banco de dados).

Personality Node (Output Formatter): Atua como o filtro final da arquitetura. Independentemente de qual agente gerou a resposta bruta técnica, este nó unifica e formata o texto para garantir o tom de voz educado, amigável e profissional da Getnet.

## 4. Pipeline RAG (Retrieval-Augmented Generation)
Para garantir que o LLM responda com dados verídicos e atualizados da empresa, implementamos um RAG funcional lendo a URL oficial em tempo de execução.

Ingestion: O WebBaseLoader raspa o HTML puro de https://www.getnet.net/en/ durante o startup da aplicação. Foi injetado também um contexto estático suplementar.

Storage & Splitting: O conteúdo é quebrado em partes semânticas usando o RecursiveCharacterTextSplitter (1000 caracteres, 100 de overlap) e vetorizado através do modelo local all-MiniLM-L6-v2 do HuggingFace. Os vetores residem na memória usando FAISS.

Retrieval: Através da ferramenta getnet_knowledge_tool, a query do usuário é vetorizada e o FAISS calcula a similaridade de cosseno, retornando os 3 chunks mais relevantes.

Generation: O LLM do Groq recebe os fragmentos contextuais injetados no prompt da tool e sintetiza a resposta final de forma embasada.

## 5. Estratégia de Testes e Qualidade (Visão de Produção)
Atendendo ao requisito de descrever a abordagem de testes de integração, detalho abaixo a estratégia arquitetural projetada para um ambiente de CI/CD real. (Nota: Para o escopo deste MVP entregue no limite de tempo, a validação foi executada via testes funcionais manuais no endpoint).

Testes de Roteamento Determinístico (Proposto): Utilização do framework pytest integrado ao TestClient do FastAPI para submeter diferentes prompts. O objetivo do teste unitário aqui não é avaliar a resposta em texto livre, mas sim afirmar (assert) estritamente se a classificação Zero-Shot do Router Agent acionou a função do especialista correto, validando a árvore de decisão.

LLM Mocking e Observabilidade (Proposto): Em esteiras automatizadas, as requisições de rede para as APIs do Groq e Tavily seriam simuladas (mockadas) utilizando pacotes como unittest.mock ou responses. Isso previne a flutuação de resultados (flakiness) por instabilidade externa e zera os custos de tokens na esteira de CI. Em produção, a observabilidade seria garantida via integração com o LangSmith.

Testes Unitários de Ferramentas (Proposto): Ferramentas customizadas (como as consultas simuladas de banco do Support Agent) demandariam testes isolados verificando o tratamento de exceções com user_ids inválidos ou vazios, blindando as ferramentas antes de serem expostas à orquestração do LLM.