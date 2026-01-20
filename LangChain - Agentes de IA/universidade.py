import json
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import pandas as pd
from typing import List

load_dotenv()

def buscar_dados_de_uma_universidade(universidade: str) -> str:
    df = pd.read_csv('universidade.csv')
    df['NOME_FACULDADE'] = df['NOME_FACULDADE'].str.lower()
    dados_com_essa_universidade = df[df['NOME_FACULDADE']==universidade]
    if dados_com_essa_universidade.empty:
        return {}
    return dados_com_essa_universidade.iloc[:1].to_dict()

def buscar_dados_das_universidades() -> str:
    df = pd.read_csv('universidade.csv')
    return df.to_dict()

    
class ExtratorDeUniversidade(BaseModel):
    universidade: str = Field(description='Nome da universidade informada, sempre em letras minusculas.')

class DadosDeUniversidade(BaseTool):
    name: str = "DadosDeUniversidade"
    description: str = "Essa ferramenta extrai os dados de uma universidade. Forneça o nome da universidade para obter os dados."
    
    def _run(self, input: str) -> str:
        # Modelo correto - gpt-4o-mini ou gpt-3.5-turbo
        llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
        parser = JsonOutputParser(pydantic_object=ExtratorDeUniversidade)
        template = PromptTemplate(
            template="""Você deve analisar e extrair o nome da universidade informada:
entrada:
-----
{input}
-----
Formato de saída: 
{formate_saida}""",
            input_variables=["input"],
            partial_variables={"formate_saida": parser.get_format_instructions()}
        )

        cadeia = template | llm | parser
        resposta = cadeia.invoke({"input": input})
        universidade = resposta
        universidade = universidade.lower().strip()
        dados = buscar_dados_de_uma_universidade(universidade)
        return json.dumps(dados)
    
class ExtratorDeTodasUniversidades(BaseTool):
    name: str = "TodasUniversidades"
    description: str = "Essa ferramenta extrai/carrega os dados de todas as universidades disponíveis. Não é necessário nenhum parametro de entrada."
    
    def _run(self, input: str) -> str:
        dados = buscar_dados_das_universidades()
        return json.dumps(dados)
