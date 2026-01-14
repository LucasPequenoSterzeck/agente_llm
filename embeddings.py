####

'''
Embeddings são representações numéricas de texto que permitem medir quão relacionados estão os diferentes pedaços de um documento.
Eles são fundamentais em diversas aplicações de processamento de linguagem natural (PLN), como busca por relevância,
agrupamento de textos por similaridade, sistemas de recomendação, detecção de anomalias, medição de diversidade textual e classificação automática de conteúdo.

Um gerador de embedding transforma uma string de texto em um vetor de números de ponto flutuante.
A distância (geralmente medida por similaridade de cosseno ou distância euclidiana) entre dois vetores de embedding
reflete quão relacionados ou semelhantes são os textos originais.

Textos com significados ou contextos similares resultarão em vetores próximos uns dos outros no espaço de embeddings,
enquanto textos com pouca ou nenhuma relação terão vetores distantes.

Os modelos text-embedding-3-small e text-embedding-3-large são os mais recentes e performáticos, oferecendo custos menores,
melhor desempenho multilíngue e novos parâmetros para controlar o tamanho geral.

Para compreender melhor, imagine uma coleção de quadrinhos em que cada item possui uma descrição textual.
Utilizando embeddings, podemos mapear essas descrições em vetores numéricos para facilitar a busca por quadrinhos
similares ou recomendar novos quadrinhos baseados em preferências de pessoas usuárias.
'''

from openai import OpenAI
client = OpenAI()

# Supondo que "descrição do quadrinho" seja a string de texto da descrição
descricao_quadrinho = "Aventuras épicas no espaço com heróis e vilões."

response = client.embeddings.create(
    input=descricao_quadrinho,
    model="text-embedding-3-small"
)

print(response.data[0].embedding)

# Resposta de exemplo para um embedding

'''
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [
        -0.006929283495992422,
        -0.005336422007530928,
        ... (omitido para brevidade)
        -4.547132266452536e-05,
        -0.024047505110502243
      ]
    }
  ],
  "model": "text-embedding-3-small",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
'''


'''
O vetor de embedding resultante pode ser extraído, salvado em um banco de dados vetorial e utilizado
 para diferentes casos de uso, como busca por relevância ou recomendações baseadas em similaridade textual.

Como podemos notar, embeddings são uma ferramenta poderosa para medir quão relacionadas estão duas 
strings e habilitar uma ampla gama de aplicações de IA.

Com os novos modelos de embedding da OpenAI, pessoas desenvolvedoras e pesquisadoras têm acesso a
ferramentas mais eficientes e multilíngues para transformar texto em números, facilitando tarefas como pesquisa e recomendação em suas coleções de dados.
'''