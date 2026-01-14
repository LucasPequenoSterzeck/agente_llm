# LCEL

"""LangChain Expression Language (LCEL) oferece uma maneira declarativa e eficiente de compor cadeias de processamento no LangChain.
LCEL é projetado para facilitar a transição de protótipos para produção, lidando com cadeias simples e complexas."""

'''Características principais do LCEL
Suporte a streaming: melhora o tempo até a primeira saída, sendo ideal para processamento em tempo real.
Suporte assíncrono: permite execução tanto síncrona quanto assíncrona, adequada para prototipagem e produção.
Execução paralela otimizada: executa etapas paralelas automaticamente para reduzir a latência.
Retentativas e alternativas: melhora a confiabilidade em escala com configurações de retentativa e alternativa.
Acesso a resultados intermediários: permite monitoramento e depuração em cadeias complexas.
Esquemas de entrada e saída: facilita a validação com esquemas Pydantic e JSONSchema.
Integração com Langsmith e Langserve: oferece observabilidade e facilita a implantação.'''

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4-0125-preview",
    temperature=0.5,
    api_key=os.getenv("OPENAI_API_KEY"))

parte1 = PromptTemplate.from_template("Analisar a queixa: {queixa}") | llm | StrOutputParser()
parte2 = PromptTemplate.from_template("Avaliar sentimento da queixa: {resultado_analise}") | llm | StrOutputParser()
parte3 = PromptTemplate.from_template("Formular resposta: {sentimento}") | llm | StrOutputParser()

cadeia = (
    {"queixa": RunnablePassthrough()}
    | RunnablePassthrough.assign(resultado_analise=parte1)
    | RunnablePassthrough.assign(sentimento=parte2)
    | parte3
)

queixa_texto = "Hoje comprei um telefone novo, modelo X com 256 GB e flip. No entanto, o produto apresentou defeito na dobradiça e não permanece fechado. O suporte não me atende e estou super arrependido."
resultado = cadeia.invoke({"queixa": queixa_texto})

print(resultado)