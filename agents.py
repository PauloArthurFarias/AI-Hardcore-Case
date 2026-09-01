"""
Módulo responsável pela orquestração do Sistema Multi-Agentes.
Implementa o padrão de Roteamento Semântico (Semantic Routing) para direcionar
as requisições de clientes para agentes especialistas (Knowledge ou Support),
garantindo eficiência de tokens, menor latência e mitigação de alucinações.
"""
import os
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from tools import search_tool, getnet_knowledge_tool, check_machine_status, check_sales_balance

# 1. INSTANCIAÇÃO DO MODELO (CÉREBRO PRINCIPAL)
# Utilizamos temperature=0 para garantir respostas determinísticas e precisas.
# Em contextos financeiros e de suporte técnico, isso é crucial para evitar alucinações criativas.
llm = ChatGroq(
    temperature=0, 
    model_name="qwen/qwen3.8-27b", # (Ou o modelo Groq que preferir)
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# 2. AGENTES ESPECIALISTAS (WORKERS)
# Agente focado em recuperação de informação corporativa (RAG) e buscas gerais na web.
knowledge_agent = initialize_agent(
    tools=[getnet_knowledge_tool, search_tool],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, # Suporta inputs estruturados (ex: objetos JSON) para as tools
    verbose=True, # Habilita logs de rastreabilidade (Thought/Action/Observation) no console
    handle_parsing_errors=True, # Resiliência: permite que o agente se recupere caso gere um JSON malformado na chamada da tool
    max_iterations=4 # Trava de segurança (Circuit Breaker) para evitar loops infinitos e custo excessivo de API
)

# Agente focado na resolução de problemas sistêmicos do cliente (acesso a dados transacionais/máquinas).
support_agent = initialize_agent(
    tools=[check_machine_status, check_sales_balance],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=4
)

# 3. PERSONALITY NODE (OUTPUT FORMATTER)
def apply_personality(raw_response: str) -> str:
    """
    Representa o nó 'Personality' do diagrama arquitetural. 
    Desacopla a execução técnica (feita pelos agentes especialistas) da formatação da interface com o usuário.
    Garante que a resposta final tenha um tom amigável, empático e padronizado da Getnet.
    """
    personality_prompt = f"""
    You are the final output formatter for Getnet customer service.
    Take the raw technical response below and rewrite it in a polite, 
    helpful, and professional tone in Portuguese. 
    If the response is already good, just refine the tone.
    
    Raw response: {raw_response}
    """
    # Invocação direta e rápida, sem passar por loops de raciocínio (tools)
    return llm.invoke(personality_prompt).content

# 4. ROUTER AGENT (ENTRY POINT)
def router_agent(user_message: str, user_id: str) -> str:
    """
    Atua como o orquestrador principal (Semantic Router).
    Avalia a intenção da mensagem (zero-shot classification) e delega a execução
    ao agente correto, em vez de sobrecarregar um único agente monolítico com dezenas de ferramentas.
    """
    router_prompt = f"""
    You are a classification router for Getnet customer service.
    Analyze the user message and classify it into ONE of these two categories:
    1. SUPPORT: For technical issues, machine errors, connectivity, or sales balance/deposits.
    2. KNOWLEDGE: For general questions about Getnet products, rates, WhatsApp link, or general web searches (like weather/currency).
    
    Respond ONLY with the exact word 'SUPPORT' or 'KNOWLEDGE'.
    
    User message: {user_message}
    """
    
    # O Roteador toma a decisão de forma objetiva, gastando o mínimo de tokens possíveis
    decision = llm.invoke(router_prompt).content.strip().upper()
    
    # Redirecionamento condicional para o worker especialista (Pattern: Supervisor/Worker)
    if "KNOWLEDGE" in decision:
        print("-> Roteado para KNOWLEDGE AGENT")
        raw_result = knowledge_agent.run(user_message)
    else:
        print("-> Roteado para SUPPORT AGENT")
        # Injeção de contexto: passamos o user_id apenas para o agente que realmente precisa consultar dados sensíveis
        raw_result = support_agent.run(f"User ID: {user_id}. Message: {user_message}")
        
    # Passa a resposta bruta pelo filtro de personalidade antes de retornar o JSON para a API (Gateway final)
    print("-> Aplicando Personality Node")
    return apply_personality(raw_result)