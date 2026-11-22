# Classificação de Mamografias

## 1. Sobre a aplicação

Aplicação web experimental para demonstração de modelos de aprendizado profundo aplicados à classificação de imagens mamográficas.

Permite o envio de imagens por meio de uma interface web, que se comunica com um backend responsável pelo processamento e pela classificação utilizando um modelo desenvolvido em PyTorch.

### Repositórios e serviços

- **Aplicação web:** https://huggingface.co/spaces/beatriz-paz/classificacao-mamografias
- **Modelo de classificação:** https://huggingface.co/beatriz-paz/model-mamografia
- **API (Render):** https://tcc-eng-telecom.onrender.com/docs

## 2. Arquitetura da aplicação

A aplicação utiliza uma arquitetura cliente-servidor, dividida em três componentes:

| Componente | Tecnologia | Responsabilidade |
|---|---|---|
| Frontend | HTML, CSS e JavaScript | Interface e envio das imagens. |
| Backend | Python e FastAPI | Recebimento e processamento das requisições. |
| Modelo de IA | PyTorch | Classificação das imagens mamográficas. |

### Fluxo de comunicação

1. O usuário acessa a interface no Hugging Face Spaces.
2. Seleciona e envia uma imagem de mamografia.
3. O JavaScript envia a imagem por uma requisição HTTP POST.
4. A API FastAPI, hospedada no Render, recebe a imagem.
5. O backend realiza o processamento e a inferência com o modelo.
6. A API retorna os resultados em JSON.
7. A interface apresenta o resultado ao usuário.

## 3. Estrutura do backend

```text
backend_mamografia/
│
├── main.py
├── requirements.txt
└── README.md
```

- **main.py:** implementação da API, endpoints, configuração de CORS e processamento das requisições.
- **requirements.txt:** dependências Python do projeto.
- **README.md:** documentação do backend.

## 4. Tecnologias e ferramentas

| Tecnologia | Utilização |
|---|---|
| Python | Desenvolvimento do backend. |
| FastAPI | Construção da API REST. |
| Uvicorn | Servidor ASGI da aplicação. |
| PyTorch | Execução do modelo de classificação. |
| HTML, CSS e JavaScript | Desenvolvimento do frontend. |
| Git e GitHub | Controle de versão e código-fonte. |
| Render | Hospedagem do backend. |
| Hugging Face Spaces | Hospedagem do frontend. |

## 5. API e endpoints

A comunicação entre frontend e backend ocorre por meio de uma API REST desenvolvida com FastAPI.

As imagens são enviadas utilizando `multipart/form-data`, e as respostas são retornadas no formato JSON.

| Método | Endpoint | Descrição |
|---|---|---|
| GET | `/` | Verifica a disponibilidade da API. |
| GET | `/health` | Verifica o estado do serviço. |
| POST | `/predict` | Recebe a imagem para processamento e classificação. |

**Documentação interativa:** https://tcc-eng-telecom.onrender.com/docs

## 6. Hospedagem

**Backend — Render**

Hospedagem da API por meio de um Web Service conectado ao GitHub.

- Root Directory: `backend_mamografia`
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

O serviço disponibiliza um endereço público para receber as requisições da interface. O plano gratuito possui limitações de recursos e pode suspender temporariamente o serviço por inatividade.

**Frontend — Hugging Face Spaces**

Interface publicada em um Space do tipo Static, utilizando HTML, CSS e JavaScript. O arquivo `script.js` configura o endereço da API para realizar o envio das imagens.

**Modelo — Hugging Face**

Os arquivos do modelo treinado estão armazenados no Hugging Face, permitindo sua disponibilização para utilização pelo backend.

## 7. Finalidade e limitações

A aplicação possui finalidade exclusivamente educacional e experimental, sendo desenvolvida no contexto de um projeto acadêmico.

Não constitui uma ferramenta de diagnóstico médico ou de análise clínica oficial. Os resultados não devem ser utilizados para decisões médicas, substituir a avaliação de profissionais de saúde ou orientar tratamentos.