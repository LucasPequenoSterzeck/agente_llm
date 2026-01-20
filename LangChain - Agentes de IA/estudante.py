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

def buscar_dados_estudante(estudante: str) -> str:
    df = pd.read_csv('estudantes.csv')
    dados_com_esse_estudante = df[df['USUARIO']==estudante]
    if dados_com_esse_estudante.empty:
        return {}
    return dados_com_esse_estudante.iloc[:1].to_dict()

class ExtratorDeEstudante(BaseModel):
    estudante: str = Field(description='Nome do estudante informado, sempre em letras minusculas. Exemplos: ana, joao, pedro, carlos.')

class DadosDeEstudante(BaseTool):
    name: str = "DadosDeEstudante"
    description: str = "Essa ferramenta extrai o histórico e preferências de um estudante. Forneça o nome do estudante para obter os dados."
    
    def _run(self, input: str) -> str:
        # Modelo correto - gpt-4o-mini ou gpt-3.5-turbo
        #llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
        #parser = JsonOutputParser(pydantic_object=ExtratorDeEstudante)
        '''template = PromptTemplate(
            template="""Você deve analisar o {input} e extrair o nome do usuário informado
                       Formato de saída: 
                       {formate_saida}""",
            input_variables=["input"],
            partial_variables={"formate_saida": parser.get_format_instructions()}
        )

        cadeia = template | llm | parser
        resposta = cadeia.invoke({"input": input})'''
        estudante = input
        estudante = estudante.lower()
        dados = buscar_dados_estudante(estudante)
        return json.dumps(dados)

# Perfil academico

class Nota(BaseModel):
    area: str = Field(description='Nome da disciplina ou área de conhecimento.')
    nota: float = Field(description='Nota obtida pelo estudante na disciplina ou área de conhecimento.')

class PerfilAcademicoDeEstudante(BaseModel):
    nome: str = Field(description='Nome completo do estudante.')
    ano_de_conclusao: str = Field(description='Ano de conclusão do ensino médio do estudante.')
    notas:List[Nota] = Field(description='Lista de notas das disciplinas e áreas de conhecimento.')
    resumo: str = Field(description='Resumo das características do estudante de forma a torna-lo único e um ótimo potencial estudante para faculdades. Exemplo: Só esse estudante tem bla bla bla.')
    '''idade: int = Field(description='Idade do estudante.')
    interesses: list[str] = Field(description='Lista de interesses acadêmicos do estudante.')
    universidades_sugeridas: list[str] = Field(description='Lista de universidades sugeridas com base nos interesses do estudante.')
    cursos_compativeis: list[str] = Field(description='Lista de cursos compatíveis com os interesses do estudante.')
    perfil_do_aluno: str = Field(description='Descrição detalhada do perfil acadêmico do estudante.')'''

class PerfilAcademico(BaseTool):
    name: str = 'PerfilAcademico'
    description: str = '''Cria um perfil academico baseado nos dados do estudante.
    Esta ferramenta requer como entrada os dados do estudante'''

    def _run(self, input: str) -> str:
        llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
        parser = JsonOutputParser(pydantic_object=PerfilAcademicoDeEstudante)
        template = PromptTemplate(template='''Formate o estudante para ser perfil academico.
- Com os dados, identifique as opções de universidades sugeridas e cursos compactiveis com o interesse do aluno.
- Destaque o perfil do aluno, dado enfase no perfil do aluno focando naquilo que faz sentido nas universidades de interesse do aluno.
Você tem que buscar os dados de estudante antes de me invocar!

Persona: Você é uma consultoria de carrera e precisa indicar com detalhes e riqueza mas direito ao ponto para o estudante as opções e consequências possíveis.
Informações atuais: 
                                  {dados_do_estudante}
                                  {formato_de_saida}''', 
                                    input_variables=['dados_do_estudante'],
                                    partial_variables={'formato_de_saida': parser.get_format_instructions()})
        
        cadeia = template | llm | parser
        resposta = cadeia.invoke({'dados_do_estudante': input})
        return resposta