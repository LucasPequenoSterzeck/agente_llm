from langchain_openai import ChatOpenAI
from langchain import hub
from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent, create_agent_executor

from estudante import DadosDeEstudante, PerfilAcademico
from universidade import DadosDeUniversidade, ExtratorDeTodasUniversidades
load_dotenv()

'''
ReAct é uma metodologia avançada que integra as habilidades de raciocínio e ação em modelos 
de linguagem para criar interações mais dinâmicas e precisas. Essa abordagem permite que os modelos
não só compreendam questões complexas, mas também interajam ativamente com informações e ambientes externos.

## Abordagem e metodologia
O framework ReAct promove uma fusão entre raciocínio detalhado e ações práticas dentro de 
um fluxo de trabalho interativo, permitindo que modelos de linguagem atuem de forma adaptativa e contextual.
Esse método é particularmente valioso em domínios que exigem verificação de fatos e respostas informadas
por dados atualizados.
'''

class AgenteOpenAIFunctions:
    def __init__(self):
        ''''O Agente sempre será uma combinação de Prompt + LLM + Ferramentas'''
        llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
        
        dados_de_estudante = DadosDeEstudante()
        perfil_academico = PerfilAcademico()
        dados_de_universidade = DadosDeUniversidade()
        todas_universidades = ExtratorDeTodasUniversidades()

        # Em tools existe a possibilidade de dar um return_dicrct=True para retornar diretamente a resposta da função, finalizando o fluxo do agente.
        self.tools = [dados_de_estudante, perfil_academico, dados_de_universidade, todas_universidades]

        # Corrigido o typo: promtp -> prompt
        self.prompt = hub.pull("hwchase17/openai-functions-agent") 
        # hwchase17/react <--- Se não for OpenAI Functions
        ## ReAct: Synergizing Reasoning and Acting in Language Models -- https://arxiv.org/abs/2210.03629

        # Criação do agente
        self.agente = create_tool_calling_agent(llm, self.tools, self.prompt) #create_agent_executor
