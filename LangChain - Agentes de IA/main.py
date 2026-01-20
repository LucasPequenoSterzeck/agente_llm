# https://github.com/alura-cursos/3860-langchain-agentes-python/tree/aula03
from dotenv import load_dotenv
from langchain.agents import AgentExecutor

from estudante import DadosDeEstudante
from agente import AgenteOpenAIFunctions

load_dotenv()

dados_de_estudante = DadosDeEstudante()
agente = AgenteOpenAIFunctions() 

pergunta = "Qual os dados da Ana?"    
pergunta = 'Crie um perfil academico para a Bianca'

  

agente_executor = AgentExecutor(agent=agente.agente, tools=agente.tools, verbose=True)

# Execução
resultado = agente_executor.invoke({"input": pergunta})
print(resultado)