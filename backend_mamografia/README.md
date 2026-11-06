# Backend — Classificação de Mamografias

## 1. Sobre o projeto

Este repositório contém o backend de uma aplicação web experimental para demonstração de modelos de aprendizado profundo aplicados à classificação de imagens mamográficas.

A aplicação foi desenvolvida para permitir que um usuário envie uma imagem de mamografia por meio de uma interface web e receba a classificação realizada por um modelo treinado em PyTorch.

O projeto está sendo desenvolvido em etapas, mantendo a interface, a API e os modelos de aprendizado profundo separados.

**Status atual:** a interface web está publicada no Hugging Face Spaces e a API FastAPI está hospedada no Render. A comunicação entre ambas foi testada com sucesso. A integração dos pesos do modelo e a classificação real ainda estão pendentes.

**Acesse aqui:** https://huggingface.co/spaces/beatriz-paz/classificacao-mamografias

## 2. Arquitetura da aplicação

A aplicação utiliza três componentes principais:

| Componente | Tecnologia | Responsabilidade |
|---|---|---|
| Frontend | HTML, CSS e JavaScript | Interface, seleção e envio da imagem |
| Backend | Python e FastAPI | Recebimento das imagens e processamento das requisições |
| Hospedagem do backend | Render | Disponibilização pública da API |
| Hospedagem do frontend | Hugging Face Spaces | Publicação da interface web |
| Modelo de IA | PyTorch | Classificação das mamografias (integração futura) |

### Fluxo de comunicação

1. O usuário acessa a interface publicada no Hugging Face Spaces.
2. Seleciona uma imagem de mamografia.
3. A interface apresenta uma pré-visualização da imagem.
4. Ao clicar em "Analisar imagem", o JavaScript envia o arquivo por meio de uma requisição HTTP POST.
5. A API FastAPI, hospedada no Render, recebe o arquivo.
6. A API retorna uma resposta em formato JSON.
7. A interface apresenta a resposta recebida.

Na implementação atual, a API confirma o recebimento da imagem. A classificação ainda não é executada.

## 3. Estrutura do backend

```text
backend_mamografia/
│
├── main.py
├── requirements.txt
└── README.md
```

### Descrição dos arquivos

- `main.py`: implementação da API FastAPI, incluindo os endpoints e a configuração de CORS.
- `requirements.txt`: dependências Python necessárias para executar o backend.
- `README.md`: documentação da aplicação e das instruções de execução.

## 4. Tecnologias utilizadas

- **Python:** linguagem utilizada no backend.
- **FastAPI:** framework para criação da API HTTP.
- **Uvicorn:** servidor ASGI utilizado para executar a aplicação.
- **python-multipart:** suporte ao recebimento de arquivos enviados por formulários.
- **HTML, CSS e JavaScript:** tecnologias utilizadas no frontend.
- **PyTorch:** tecnologia prevista para execução do modelo de classificação.

## 5. Endpoints disponíveis

| Método | Endpoint | Descrição |
|---|---|---|
| GET | `/` | Verifica se a API está funcionando |
| GET | `/health` | Verifica o estado do serviço |
| POST | `/predict` | Recebe uma imagem para análise |

### POST /predict

Recebe uma imagem por meio de `multipart/form-data`, utilizando o campo `file`.

Exemplo de resposta atual:

```json
{
  "status": "success",
  "filename": "mamografia.png",
  "message": "Imagem recebida pela API com sucesso!",
  "image_size_bytes": 123456,
  "prediction": null
}
```

O campo `prediction` permanece `null` até a integração do modelo de aprendizado profundo.

## 6. Configuração e execução local

### Pré-requisitos

- Python 3 instalado.
- Git instalado.
- Ambiente virtual Python.

### Criar o ambiente virtual

Na raiz da pasta `backend_mamografia`, execute:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Instalar as dependências

```bash
pip install -r requirements.txt
```

### Executar a API

```bash
uvicorn main:app --reload
```

A API estará disponível localmente em:

```text
http://127.0.0.1:8000
```

A documentação interativa dos endpoints poderá ser acessada em:

```text
http://127.0.0.1:8000/docs
```

## 7. Hospedagem e publicação

### Backend — Render

O backend está hospedado no Render por meio de um Web Service conectado ao repositório GitHub.

Configurações utilizadas:

| Configuração | Valor |
|---|---|
| Root Directory | `backend_mamografia` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |

O Render disponibiliza uma URL pública para acesso à API.

### Frontend — Hugging Face Spaces

O frontend está publicado em um Space do tipo Static, utilizando HTML, CSS e JavaScript.

O arquivo `script.js` contém o endereço público da API, utilizado para enviar as imagens ao endpoint `/predict`.

## 8. Etapas de desenvolvimento

### Etapa 1 — Interface web

- [x] Criação do Space no Hugging Face.
- [x] Desenvolvimento do arquivo `index.html`.
- [x] Desenvolvimento do arquivo `style.css`.
- [x] Desenvolvimento do arquivo `script.js`.
- [x] Implementação da seleção de imagens.
- [x] Implementação da pré-visualização da imagem.
- [x] Teste de funcionamento da interface.

### Etapa 2 — Backend e comunicação

- [x] Criação da API utilizando FastAPI.
- [x] Implementação do endpoint `/`.
- [x] Implementação do endpoint `/health`.
- [x] Implementação do endpoint `/predict` para recebimento de imagens.
- [x] Configuração de CORS.
- [x] Criação do repositório e da pasta do backend no GitHub.
- [x] Publicação do backend no Render.
- [x] Conexão entre o Hugging Face e o Render.
- [x] Teste de envio de imagem e recebimento da resposta pela interface.

### Etapa 3 — Integração do modelo de aprendizado profundo

**Pendente — modelo ainda em desenvolvimento.**

As próximas atividades previstas são:

- [ ] Definir a arquitetura final do modelo.
- [ ] Organizar os pesos treinados para utilização na aplicação.
- [ ] Implementar o carregamento dos pesos no backend.
- [ ] Reproduzir o pré-processamento utilizado durante o treinamento.
- [ ] Implementar a inferência com PyTorch.
- [ ] Retornar a classe prevista e as informações relevantes em JSON.
- [ ] Apresentar o resultado da classificação na interface.
- [ ] Realizar testes com imagens que não participaram do treinamento.

## 9. Considerações importantes

- A aplicação encontra-se em desenvolvimento e não deve ser utilizada para diagnóstico médico.
- A classificação real ainda não está implementada.
- A arquitetura do modelo e seus pesos serão definidos após a conclusão dos experimentos.
- O pré-processamento utilizado na inferência deverá reproduzir corretamente o procedimento adotado no treinamento.
- Imagens utilizadas na demonstração devem ser públicas e devidamente anonimizadas.
- A configuração atual de CORS permite requisições de qualquer origem. Antes de uma disponibilização mais ampla, recomenda-se restringir o acesso ao domínio do frontend.
- O plano gratuito do Render pode apresentar suspensão temporária por inatividade e limitações de recursos.

---

**Projeto:** Classificação de Mamografias  
**Finalidade:** Pesquisa e demonstração experimental de modelos de aprendizado profundo.