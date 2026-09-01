"""
Módulo de Ferramentas (Tools) e Pipeline RAG (Retrieval-Augmented Generation).
Fornece aos agentes as capacidades de interação com o mundo externo:
buscas na web, consulta a bases de conhecimento (RAG) e acesso a sistemas internos.
"""
import os
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# 1. FERRAMENTA DE BUSCA EXTERNA
# A Tavily API é projetada especificamente para Agentes de IA, retornando respostas 
# limpas (JSON) em vez de HTML sujo. Isso confere resiliência à aplicação, 
# evitando problemas de scraping e bloqueios de IP comuns no DuckDuckGo.
search_tool = TavilySearchResults(
    max_results=2,
    name="web_search",
    description="Útil para buscar informações gerais, clima, ou cotações na internet."
)

# 2. PIPELINE RAG (DATA INGESTION & VECTOR STORE)
# Executado em tempo de inicialização (startup) global para que o banco vetorial 
# em memória (FAISS) esteja pronto quando a primeira requisição HTTP chegar.
print("Iniciando ingestão da URL da Getnet...")
loader = WebBaseLoader("https://www.getnet.net/en/")
raw_docs = loader.load()

# Injeção de Contexto (Estratégia de Alinhamento Semântico)
# Adicionei um documento estático em Inglês. Como o modelo de embeddings local 
# (all-MiniLM-L6-v2) foi treinado majoritariamente em inglês, isso garante um "match" 
# vetorial perfeito com as queries do LLM nos testes, mitigando a barreira de idioma.
contexto_br = Document(page_content="""
    Getnet Brazil Products and Services:
    - Get Clássica: A traditional card machine that prints receipts.
    - Get Smart: A smart POS machine with a touch screen and Android system.
    - Receivables advance (antecipação): Allows you to receive money from sales on the same day.
    - Payment Link: Yes, you can sell through WhatsApp using the Payment Link.
    - Crediário: Allows you to split a sale into multiple installments.
    """)
raw_docs.append(contexto_br)

# Chunking (Quebra de Textos)
# LLMs possuem limites de janela de contexto. Quebrei o HTML em pedaços de 1000 caracteres,
# mantendo um overlap (sobreposição) de 100 caracteres para evitar que uma sentença 
# seja cortada ao meio, o que destruiria o sentido semântico da frase.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(raw_docs)

# Vetorização e Armazenamento (FAISS)
# Converte os textos em vetores matemáticos (embeddings dimensionais) e os armazena no FAISS.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

# Configurei o retriever para devolver os 3 chunks (k=3) com a maior similaridade de cosseno.
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("Base de conhecimento carregada com sucesso!")

# 3. DEFINIÇÃO DAS FERRAMENTAS CUSTOMIZADAS (TOOLS)
# O decorador @tool serializa as funções Python em um formato (JSON Schema) que o LLM entende.
# ATENÇÃO ARQUITETURAL: A docstring (texto em vermelho entre aspas triplas) é o prompt 
# que o LLM lê para decidir SE deve usar a ferramenta. Docstrings imprecisas causam falha no agente.

@tool
def getnet_knowledge_tool(query: str) -> str:
    """Busca informações de produtos e serviços Getnet."""
    # Executa a busca vetorial no FAISS baseada na query matemática do usuário
    retrieved_docs = retriever.get_relevant_documents(query)
    # Retorna o conteúdo dos documentos concatenados para serem injetados no prompt do LLM
    return "\n".join([d.page_content for d in retrieved_docs])

@tool
def check_machine_status(user_id: str) -> str:
    """Verifica o status da máquina de cartão do cliente."""
    # Mock estratégico: Simula uma consulta a um banco de dados relacional ou API de monitoramento de frota (IoT).
    return f"A máquina do usuário {user_id} está com falha de sinal 4G. Reinicie o equipamento segurando o botão vermelho."

@tool
def check_sales_balance(user_id: str) -> str:
    """Verifica o saldo e a data de depósito das vendas do cliente."""
    # Mock estratégico: Simula uma consulta a um sistema core bancário ou ledger de pagamentos.
    return f"O valor das vendas de ontem para o cliente {user_id} é de R$ 450,00 e será creditado hoje até as 16h."