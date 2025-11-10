# 1️⃣ RPA (UiPath, Blue Prism, Automation Anywhere, BotCity)

# Projetos práticos:

# Criar um bot que organize arquivos de uma pasta, separando por tipo (.txt, .csv, .pdf).


# import os
# import shutil

# def organizar_pasta(diretorio):
#     if not os.path.exists(diretorio):
#         print('diretório não encontrado!')
#         return

#     extensoes = {
#         '.txt': 'Textos',
#         '.csv': 'Planilhas',
#         '.pdf': 'PDFs'
#     }

#     for arquivo in  os.listdir(diretorio):
#         caminho_arquivo = os.path.join(diretorio, arquivo)

#         if os.path.isdir(caminho_arquivo):
#             continue

#         _, ext = os.path.splitext(arquivo)

#         if ext.lower() in extensoes:
#             pasta_destino = os.path.join(diretorio,extensoes[ext.lower()])

#             os.makedirs(pasta_destino, exist_ok=True)

#             novo_caminho = os.path.join(pasta_destino, arquivo)
#             shutil.move(caminho_arquivo, novo_caminho)
#             print(f'Movido: {arquivo} >> {pasta_destino}')

#     print('Organização concluida')


# if __name__ == '__main__':
#     pasta = r'C:\Users\Maxwell Fernandes\Downloads'
#     organizar_pasta(pasta)


# Criar um bot que leia planilhas, filtre dados específicos e gere relatórios em Excel/CSV.

#CRIANDO PRIMEIRAMENTE UMA PLANILHA


# import pandas as pd
# import random

# def gerar_planilha_teste(nome='dados.xlsx', linhas=1000):
#     nomes = ['Ana', 'Carlos', 'Maria', 'João', 'Paula', 'Marcos', 'Juliana'
#              , 'Pedro']
#     cidades = ['São paulo', 'Rio de janeiro', 'Belo horizonte', 'salvador,'
#     'Fortaleza', 'Curitiba']

#     dados = {
#         'Nome': [random.choice(nomes) for _ in range(linhas)],
#         'Cidade': [random.choice(cidades) for _ in range(linhas)],
#         'Idade': [random.randint(18, 70) for _ in range(linhas)],
#         'Salário': [round(random.uniform(2000, 15000), 2) for _ in range(linhas)]
#     }
    
#     df = pd.DataFrame(dados)

#     df.to_excel(nome, index = False)
#     df.to_csv(nome.replace('.xlsx', '.csv'), index=False)

#     print(f'Planilha gerada com {linhas} linhas: {nome} e {nome.replace('.xlsx', '.csv')}')

# if __name__ == '__main__':
#     gerar_planilha_teste('dados.xlsx', linhas=5000)


# criando o BOT que filtra e cria um outro arquivo tanto XLSX e CSV


# import pandas as pd
# import os

# def gerar_relatorio(caminho_planilha, coluna_filtro, valor_filtro, nome_saida='relatiorio.csv'):
#     if not os.path.exists(caminho_planilha):
#         print(f'Erro: arquivo {caminho_planilha} não encontrado')
#         return
    
#     if caminho_planilha.endswith('.csv'):
#         df = pd.read_csv(caminho_planilha)
#     else:
#         df = pd.read_excel(caminho_planilha)

#     print('Planilha carregada com sucesso!')

#     if coluna_filtro not in df.columns:
#         print(f'Erro: coluna "{coluna_filtro}" não existe na planilha.')
#         return
    
#     df_filtrado = df[df[coluna_filtro] == valor_filtro]

#     df_filtrado.to_csv(nome_saida, index=False)
#     df_filtrado.to_excel(nome_saida.replace('.csv', '.xlsx'), index=False)

#     print(f'Relatórios gerados: {nome_saida} e {nome_saida.replace('.csv','.xlsx')}')

# if __name__ == '__main__':
#     gerar_relatorio('dados.xlsx', coluna_filtro='Cidade', valor_filtro='São paulo')


# Criar um bot que envie e-mails automáticos com anexos selecionados.

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Carrega modelo e tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Conversa
chat_history_ids = None

print("Chatbot: Olá! Me pergunte algo. (digite 'sair' para encerrar)")
while True:
    user_input = input("Você: ")
    if user_input.lower() == "sair":
        break

    # Codifica a entrada do usuário
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Concatena histórico de conversas
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

    # Gera resposta
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decodifica e exibe resposta
    resposta = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print("Chatbot:", resposta)



# Criar um bot que interaja com um site (login, coleta de dados e download).

# Questões de fixação:

# Como tratar exceções (arquivo não encontrado, planilha vazia) em bots?

# Como registrar logs de execução e falhas?

# Como otimizar bots para reduzir tempo de execução e evitar loops desnecessários?

# 🔁 Repita esses exercícios em duas ferramentas diferentes de RPA para fixação.

# 2️⃣ Agentes de IA / Low-Code Workflow (n8n, Abacus.AI, Manus AI)

# Projetos práticos:

# Criar um agente que receba perguntas em texto e gere respostas via API de IA (OpenAI ou Claude).

# Criar um agente que resuma textos longos automaticamente e salve em CSV.

# Criar um fluxo n8n que combine dados de APIs externas e planilhas para gerar relatórios automatizados.

# Questões de fixação:

# Como integrar múltiplas APIs em um mesmo fluxo?

# Como lidar com entradas inválidas ou ausentes?

# Como monitorar a execução e registrar falhas?

# 3️⃣ Python (Integrações e funções avançadas)

# Projetos práticos:

# Criar scripts que leiam CSV/JSON, filtrem dados e enviem via API REST.

# Criar funções que validem e-mails ou números de telefone usando regex.

# Criar pipelines simples que combinem dados de RPA + agentes de IA + logs em um arquivo final.

# Automatizar alertas por e-mail sempre que uma automação falhar.

# Questões de fixação:

# Como modularizar código para facilitar manutenção?

# Como tratar erros de conexão em APIs externas?

# Como otimizar leitura e escrita de grandes arquivos CSV/JSON?

# 4️⃣ Infraestrutura / Cloud / Virtualização (AWS, Azure, GCP, VMware, Hyper-V, Bash, PowerShell)

# Projetos práticos:

# Criar uma VM Linux local ou em cloud (VirtualBox/VMware/AWS EC2).

# Criar container Docker com Python + script de automação e persistência de logs.

# Criar script Bash ou PowerShell para backup de arquivos e recuperação de dados.

# Criar monitoramento simples: checar se serviço/VM/container está ativo e enviar alerta por e-mail.

# Questões de fixação:

# Qual diferença entre VM e container?

# Como automatizar backups e testes de recuperação?

# Como medir performance e capacidade de servidores ou containers?

# 5️⃣ LLMs / RAG / Langchain / LangGraph / LanceDB / IA Generativa

# Projetos práticos:

# Criar pipeline RAG:

# Recebe pergunta

# Busca informações em documentos ou banco vetorial (LanceDB)

# Retorna resposta baseada em contexto

# Criar integração com múltiplas LLMs (OpenAI, Claude, Mistral) para comparar respostas.

# Criar fluxo Langchain/Graph que:

# Processa dados

# Aplica filtros ou transformações

# Gera saídas em CSV/JSON

# Treinar modelo simples (TensorFlow / PyTorch / Keras) para classificação de texto ou análise de sentimento.

# Questões de fixação:

# Qual a diferença entre embeddings e respostas diretas de LLM?

# Como integrar bancos vetoriais em pipelines de IA?

# Como versionar e testar fluxos de IA complexos?

# 6️⃣ Bancos de dados (SQL e NoSQL)

# Projetos práticos:

# Criar banco relacional (PostgreSQL / MySQL / Oracle) para armazenar dados de bots ou agentes.

# Criar banco não-relacional (MongoDB / Cassandra) para logs ou resultados de IA.

# Criar banco vetorial (LanceDB) para RAG/embeddings.

# Criar scripts Python para leitura e escrita nos bancos de dados, integrando com RPA ou IA.

# Questões de fixação:

# Quando usar SQL vs NoSQL vs banco vetorial?

# Como otimizar consultas para grandes volumes de dados?

# Como garantir consistência e backups dos dados?

# 7️⃣ Metodologias ágeis e Soft Skills (Scrum / Kanban / SRE / DevOps)

# Projetos práticos:

# Criar kanban board no Trello ou Jira para organizar tarefas de automação e IA.

# Simular daily meeting: cada “bot” ou agente é uma tarefa com status, impedimentos e próximos passos.

# Documentar processos e criar README para cada automação/projeto.

# Criar postmortem de falhas simuladas de bots ou pipelines de IA.

# Questões de fixação:

# Como priorizar tarefas usando Scrum/Kanban?

# Como identificar causa raiz de falhas em automações ou IA?

# Como comunicar problemas técnicos para não técnicos?

# 8️⃣ Repetição e fixação

# Para fixação, repita exercícios em diferentes combinações:

# Bot RPA + Python + banco de dados

# Agente de IA + RAG + Langchain + LanceDB

# Scripts de backup + container + monitoramento

# A ideia é que você consiga fazer um fluxo completo que percorra: coleta de dados → processamento → armazenamento → relatório → monitoramento → logs → postmortem.

# # # # Bloco 1 – Linux & comandos básicos

# # # # Crie um diretório chamado projeto_vaga e dentro dele crie subpastas logs, scripts, backup.

# # # # Liste todos os arquivos dentro de projeto_vaga e filtre apenas os arquivos .log.

# # # # Redirecione a saída de um comando (ls -l) para um arquivo chamado saida.txt.

# # # # Use grep para buscar a palavra “ERRO” dentro de todos os arquivos .log da pasta logs.

# # # # Crie um script backup.sh que copie todos os arquivos .log da pasta logs para a pasta backup e adicione a data no nome do arquivo.

# # # # Bloco 2 – Python e automação

# # # # Crie um script Python que leia todos os arquivos .log da pasta logs e conte quantas vezes cada palavra aparece.

# # # # Crie um script Python que leia os logs, filtre apenas linhas com a palavra “ERRO” e salve em um novo arquivo erros.txt.

# # # # Crie um script Python que:

# # # # Liste todos os arquivos da pasta backup

# # # # Verifique se algum arquivo tem mais de 7 dias

# # # # Apague os arquivos antigos automaticamente

# # # # Crie uma função Python que receba um JSON de usuários (nome/email) e atualize o email de um usuário específico, salvando o resultado no mesmo arquivo.

# # # # Bloco 3 – Banco de dados

# # # # Crie um banco SQLite (local, fácil de testar) com uma tabela usuarios (id, nome, email).

# # # # Insira 5 usuários nessa tabela.

# # # # Escreva um script Python que consulte todos os usuários cujo nome comece com “A”.

# # # # Escreva um script que faça backup da tabela usuarios para um arquivo CSV.

# # # # ⚠️ Para Oracle, MongoDB ou PostgreSQL, você pode simular localmente com Docker ou SQLite/MongoDB local.

# # # # Bloco 4 – Docker / Kubernetes / Microserviços

# # # # Crie um container Docker com uma aplicação Python simples (por exemplo, imprime “Hello World”).

# # # # Suba esse container localmente e veja se roda.

# # # # Escreva um arquivo YAML de Deployment e Service para rodar essa aplicação em Kubernetes local (Minikube ou Kind).

# # # # Escale a aplicação para 3 réplicas.

# # # # Simule que um pod falhou e veja se o Kubernetes reinicia outro automaticamente.

# # # # Crie um script Shell que faça deploy do YAML no Kubernetes com um único comando.

# # # # Bloco 5 – Monitoramento / ELK Stack

# # # # Crie um arquivo de logs simulando erros (ex: “ERRO: Falha no login”, “INFO: Usuário logado”).

# # # # Configure um dashboard simples no Kibana (ou visualize localmente com Python/Plotly) mostrando:

# # # # Total de logs por tipo (INFO/ERRO)

# # # # Últimos 5 erros

# # # # Escreva um script Python que leia o log e envie um alerta (print ou email) se houver mais de 5 erros seguidos.

# # # # Bloco 6 – Conceitos SRE / DevOps

# # # # Explique em um README:

# # # # O que é postmortem

# # # # Diferença entre DevOps e SRE

# # # # O que significa scalabilidade de um sistema

# # # # Crie uma checklist de troubleshooting de uma aplicação web (ex: logs, banco, container, rede, CPU/memória).

# # # # Documente no README como você automatizou backup, deploy e monitoramento no seu mini-projeto.

# # # # ✅ Observações

# # # # Todos os exercícios são práticos, como a vaga pede.

# # # # Você vai praticar Linux, Python, Shell Script, Docker, Kubernetes, logs, monitoramento e automação, tudo no VS Code.

# # # # Se você completar esses 25 exercícios, estará muito próximo do perfil que a vaga pede.

# # # # Se você quiser, posso montar esse mesmo plano em um formato “projeto completo”, tipo:

# # # # Uma aplicação de exemplo com microserviço Python, logs, deploy em Kubernetes, monitoramento e scripts automatizados, tudo para treinar e mostrar como portfólio.

# # # # Bloco 1 – n8n (Workflow / Automação visual)

# # # # Crie um workflow no n8n que receba dados de um formulário (simulado com webhook), valide o e-mail e salve em um arquivo JSON.

# # # # Crie um workflow que:

# # # # Leia dados de uma API pública (ex.: https://jsonplaceholder.typicode.com/users
# # # # )

# # # # Filtre apenas usuários com “.com” no e-mail

# # # # Salve o resultado em um arquivo local

# # # # Configure um trigger periódico (Cron) no n8n que execute um workflow a cada 5 minutos e registre a execução em logs.

# # # # Crie um workflow que envie alertas via Slack ou email sempre que um valor específico aparecer nos dados.

# # # # Bloco 2 – Integração com APIs e Webhooks

# # # # Use Python ou JavaScript para chamar uma API REST e imprimir os resultados no console.

# # # # Crie um script que:

# # # # Receba dados de uma API

# # # # Transforme os dados (ex.: converta strings em maiúsculas)

# # # # Envie os dados para outro endpoint (simulado local ou mock API)

# # # # Crie um webhook local usando Flask ou FastAPI que receba requisições JSON e retorne uma resposta customizada.

# # # # Bloco 3 – IA / LLM / NLP

# # # # Crie um workflow que use uma API de IA (OpenAI, Hugging Face ou similar) para:

# # # # Receber um texto

# # # # Resumir o conteúdo

# # # # Salvar o resumo em um arquivo JSON ou banco de dados local

# # # # Automatize um chatbot simples usando n8n e integração com GPT:

# # # # Receba a pergunta

# # # # Retorne a resposta do modelo

# # # # Salve histórico das conversas em JSON

# # # # Escreva um script Python que filtre palavras-chave em textos recebidos e envie alerta se encontrar alguma palavra específica.

# # # # Bloco 4 – Python / JavaScript para customizações

# # # # Dentro de um workflow n8n, crie um script Node em JavaScript que transforme dados de entrada (ex.: calcular média de valores).

# # # # Crie um script Python que:

# # # # Leia um arquivo JSON

# # # # Modifique valores específicos (ex.: atualizar emails ou status)

# # # # Salve novamente no mesmo arquivo

# # # # Crie funções que possam ser reutilizadas em diferentes workflows (ex.: validação de e-mail, parsing de datas, formatação de strings).

# # # # Bloco 5 – Infraestrutura e Docker

# # # # Crie um container Docker com n8n configurado e teste localmente.

# # # # Configure volumes para persistir os dados do n8n no container, garantindo que workflows não sejam perdidos ao reiniciar.

# # # # Crie um script de deploy (Shell ou Python) que suba o container com um único comando.

# # # # Simule que o container caiu e verifique se o deploy automático reinicia corretamente (health check simples).

# # # # Bloco 6 – Conceitos SRE / DevOps / Observabilidade

# # # # Crie um dashboard simples (pode ser um CSV + Plotly ou Kibana/Power BI) mostrando:

# # # # Quantidade de execuções de workflows

# # # # Número de erros por dia

# # # # Documente passo a passo de um postmortem de falha em workflow:

# # # # Qual workflow caiu

# # # # Logs coletados

# # # # Causa raiz

# # # # Solução aplicada

# # # # Faça uma checklist de troubleshooting para integrações com APIs e IA (checagem de logs, tokens, endpoints, dados).

# # # Bloco 1 – Mapeamento e otimização de processos

# # # Escolha um processo repetitivo do seu dia a dia (ex: organizar arquivos ou e-mails) e descreva passo a passo como ele poderia ser automatizado.

# # # Crie um fluxograma simples mostrando cada passo do processo que será automatizado.

# # # Bloco 2 – RPA (UiPath / Blue Prism / Automation Anywhere / BotCity)

# # # Crie um bot simples em UiPath ou BotCity que:

# # # Abra uma pasta

# # # Leia os nomes de todos os arquivos

# # # Salve a lista em um arquivo Excel ou CSV

# # # Modifique o bot para que ele filtre arquivos por extensão (ex.: .txt ou .pdf) antes de salvar.

# # # Crie um bot que envie automaticamente um e-mail com um anexo específico usando a ferramenta RPA.

# # # Bloco 3 – Agentes de IA (Abacus.AI, Manus AI, etc.)

# # # Crie um workflow de agente de IA que:

# # # Receba uma pergunta em texto

# # # Chame uma API de IA para gerar a resposta

# # # Armazene a resposta em um arquivo JSON ou banco local

# # # Modifique o agente para classificar a resposta como “sim/não/precisa de revisão” baseado em palavras-chave do texto.

# # # Automatize o envio do resultado para uma planilha ou e-mail.

# # # Bloco 4 – Python para integrações e funções avançadas

# # # Crie um script Python que:

# # # Leia dados de um arquivo CSV

# # # Filtre linhas com valores específicos

# # # Envie os dados filtrados para uma API REST fictícia

# # # Escreva uma função Python que:

# # # Receba uma lista de e-mails

# # # Valide o formato de cada e-mail (regex)

# # # Retorne apenas os e-mails válidos

# # # Crie um script que combine dados de duas fontes diferentes (ex.: CSV + JSON) e salve o resultado em um novo arquivo.

# # # Bloco 5 – Teste, monitoramento e performance

# # # Simule um workflow que falha (ex.: dados incorretos ou API fora do ar) e registre o erro em um arquivo de log usando Python.

# # # Crie um script de monitoramento que:

# # # Leia o log

# # # Envie alerta (print ou e-mail) se mais de 3 erros acontecerem consecutivamente

# # # Crie um dashboard simples (Python + Plotly ou Excel) mostrando:

# # # Número de execuções do bot/IA por dia

# # # Número de erros detectados

# # # Bloco 6 – Conceitos ágeis e inovação

# # # Crie um README explicando:

# # # O processo automatizado

# # # Qual ferramenta você usou

# # # Como ele pode ser escalado ou melhorado

# # # Descreva uma melhoria que você implementaria para tornar o bot/IA mais eficiente ou seguro.

# # Bloco 1 – Mapeamento e otimização de processos

# # Escolha um processo repetitivo do seu dia a dia (ex.: organizar arquivos, e-mails ou planilhas).

# # Descreva passo a passo cada tarefa.

# # Identifique pontos que podem ser automatizados.

# # Crie um fluxograma mostrando visualmente cada passo do processo.

# # Explique em poucas palavras como você decidiria se o processo é candidato à automação.

# # Refatore o fluxograma adicionando condições de erro e alertas para falhas.

# # 🔁 Repita esse exercício com 3 processos diferentes, para treinar análise de processos e identificar padrões de automação.

# # Bloco 2 – RPA (UiPath / Blue Prism / Automation Anywhere / BotCity)

# # Crie um bot que:

# # Abra uma pasta de arquivos

# # Leia os nomes de todos os arquivos

# # Salve em um arquivo Excel ou CSV

# # Modifique o bot para filtrar arquivos por extensão antes de salvar.

# # Crie um bot que envie automaticamente e-mails com anexo específico usando a ferramenta de RPA.

# # Crie um bot que leia dados de uma planilha, filtre linhas específicas e escreva os resultados em outra planilha.

# # Teste os bots adicionando erros intencionais (arquivo não encontrado, planilha vazia) e registre os erros em um log.

# # 🔁 Repita os exercícios 5-9 usando uma segunda ferramenta RPA, apenas para familiarização.

# # Bloco 3 – Agentes de IA (Abacus.AI, Manus AI)

# # Crie um agente de IA que:

# # Receba uma pergunta em texto

# # Use uma API de IA para gerar a resposta

# # Salve a resposta em JSON ou banco local

# # Modifique o agente para classificar a resposta como “sim/não/precisa de revisão” com base em palavras-chave.

# # Automatize o envio da resposta para uma planilha ou e-mail.

# # Crie outro agente que:

# # Receba um conjunto de textos

# # Resuma automaticamente cada texto

# # Salve os resumos em CSV

# # Teste o agente inserindo inputs inválidos (texto vazio, caracteres especiais) e registre os erros.

# # 🔁 Repita exercícios 10-14 pelo menos duas vezes, variando o tipo de input (textos longos, tabelas, e-mails).

# # Bloco 4 – Python para integrações e funções avançadas

# # Crie um script Python que:

# # Leia um CSV

# # Filtre linhas com valores específicos

# # Envie os dados filtrados para uma API REST fictícia

# # Escreva uma função Python que:

# # Receba uma lista de e-mails

# # Valide cada e-mail usando regex

# # Retorne apenas os e-mails válidos

# # Crie um script que combine dados de duas fontes diferentes (CSV + JSON) e salve em novo arquivo.

# # Crie uma função que:

# # Receba dados do RPA ou IA

# # Faça uma transformação simples (ex.: normalizar texto, calcular média de números, substituir valores nulos)

# # Automatize o envio de logs ou alertas via Python sempre que o bot ou agente de IA encontrar um erro.

# # 🔁 Repita exercícios 15-19 para diferentes conjuntos de dados e APIs para fixar.

# # Bloco 5 – Infraestrutura / Cloud / Virtualização

# # Crie uma VM local (VirtualBox ou VMware) e instale um Linux simples.

# # Crie um container Docker com Python e um script de automação simples.

# # Configure volumes e persistência no container para salvar logs ou outputs do bot/IA.

# # Simule uma falha do container ou VM e teste se o script de monitoramento detecta o problema.

# # Faça um backup simples de dados da VM ou container e teste a recuperação.

# # Crie um script Bash ou PowerShell que:

# # Leia logs

# # Gere um relatório de erros

# # Envie alerta via e-mail (simulado)

# # 🔁 Repita exercícios 20-25 para praticar infraestrutura + scripts de automação.

# # Bloco 6 – Teste, monitoramento e performance

# # Crie um log central para bots e agentes de IA que registre:

# # Hora da execução

# # Status (sucesso/falha)

# # Mensagem de erro se houver

# # Crie um dashboard simples (Python + Plotly ou Excel) mostrando:

# # Número de execuções do bot por dia

# # Quantidade de erros detectados

# # Teste a resiliência do workflow inserindo dados inválidos e analisando se o log registra corretamente.

# # Escreva um postmortem simulando a falha de um bot/IA:

# # Qual workflow falhou

# # Causa raiz

# # Ação corretiva

# # Bloco 7 – Metodologias ágeis / documentação / soft skills

# # Escreva um README explicando um processo automatizado:

# # O que ele faz

# # Ferramentas usadas

# # Como monitorar e manter

# # Crie uma tabela de melhorias possíveis para o workflow, priorizando impacto x esforço.

# # Simule uma reunião de revisão: explique para um colega como o processo funciona, problemas detectados e soluções aplicadas.

# # 🔁 Faça isso para pelo menos 3 workflows diferentes, garantindo prática em documentação e comunicação.



# 1️⃣ Vagas de RPA / Agentes de IA

# RPA:

# UiPath

# Blue Prism

# Automation Anywhere

# BotCity

# Agentes de IA / Automação Inteligente:
# 5. Abacus.AI
# 6. Manus AI

# Workflow / Integração Low-Code:
# 7. n8n

# APIs / LLM / NLP:

# Não é um “app”, mas uso de APIs externas de IA (OpenAI, Hugging Face, etc.)

# 2️⃣ Infraestrutura / Cloud / Virtualização

# Servidores / Cloud:
# 8. AWS
# 9. Azure
# 10. GCP

# Virtualização:
# 11. VMware
# 12. Hyper-V

# Ferramentas de monitoramento / dashboards:
# 13. ELK Stack (Elasticsearch, Logstash, Kibana)

# 3️⃣ Bancos de dados

# Relacionais:
# 14. Oracle
# 15. PostgreSQL

# Não-relacionais:
# 16. MongoDB
# 17. Cassandra

# 4️⃣ Outras ferramentas / conceitos

# Middleware / integração:
# 18. Weblogic Application
# 19. SOA Suite
# 20. Azure (já contado em Cloud)
# 21. Apache
# 22. OHS
# 23. Axway API Gateway

# Metodologias / certificações:

# Scrum / Kanban / SRE / DevOps (não são apps, mas relevantes)

# 5️⃣ Linguagens / scripting

# Python

# Shell script / Bash

# PowerShell

# JavaScript / TypeScript