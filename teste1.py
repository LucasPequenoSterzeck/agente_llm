# JsonOutputParser e PydanticOutputParser

'''O JsonOutputParser é particularmente útil quando a saída necessita ser mapeada em diferentes categorias ou itens.
Suporta classes Pydantic, facilitando a transformação da saída do LLM em objetos estruturados e prontos para uso em aplicações.
Isso é extremamente útil para sumarizar dados complexos, como tickets de suporte, em categorias distintas como Issue, Root Causes e Resolution.'''

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=os.getenv("OPENAI_API_KEY"))

# Defina a classe com a estrutura desejada
class Bandeira(BaseModel):
    pais: str = Field(description="nome do pais")
    cores: str = Field(description="cor principal da bandeira")
    historia: str = Field(description="história da bandeira")

# Defina o prompt que será utilizado para pergunta
flag_query = "Me fale da bandeira do Brasil"

# Defina a estrutura que será utilizada para processar a saída
parseador_bandeira = JsonOutputParser(pydantic_object=Bandeira)

prompt = PromptTemplate(
    template="Responda a pergunta do usuário.\n{instrucoes_formato}\n{pergunta}\n",
    input_variables=["pergunta"],
    partial_variables={"instrucoes_formato": parseador_bandeira.get_format_instructions()},
)

chain = prompt | llm | parseador_bandeira

chain.invoke({"pergunta": flag_query})