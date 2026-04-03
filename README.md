# 🚀 Otimização de Hiperparâmetros de CNN com Algoritmo Genético

## Visão Geral do Projeto

Este é um projeto full-stack que implementa um **Algoritmo Genético (AG)** para otimizar os hiperparâmetros de uma **Rede Neural Convolucional (CNN)**. A aplicação web permite configurar os parâmetros do AG, executar o processo de otimização no backend, visualizar o progresso em tempo real e analisar os resultados finais, incluindo exemplos de predições. Todo o sistema é containerizado usando Docker para garantir portabilidade e fácil implantação.

O projeto é ideal para entender a aplicação prática de algoritmos genéticos na otimização de modelos de Machine Learning, bem como a arquitetura de microsserviços e containerização.

## ✨ Funcionalidades

-   **Otimização Customizável:** Configure diversos hiperparâmetros do Algoritmo Genético (tamanho da população, gerações, taxa de mutação, épocas por modelo) e opções de busca para a CNN (taxas de aprendizado, tamanhos de batch, número de filtros, neurônios FC, taxas de dropout).
-   **Execução em Tempo Real:** Monitore o progresso do AG em tempo real diretamente na interface web, com atualizações por geração e por indivíduo avaliado.
-   **Logs Interativos:** Visualize logs detalhados da execução do backend em tempo real através de WebSockets.
-   **Controle de Execução:** Inicie e, opcionalmente, pare a execução do Algoritmo Genético a qualquer momento pela interface.
-   **Análise de Resultados:**
    -   Relatório final detalhado com o melhor conjunto de hiperparâmetros encontrado, acurácia e precisão média.
    -   Gráfico interativo da evolução da acurácia dos melhores indivíduos ao longo das gerações.
    -   Visualização de 5 exemplos de predições corretas e 5 exemplos de predições incorretas do modelo final.
-   **Exportação de Resultados:** Exporte os resultados finais da otimização para um arquivo JSON.
-   **Configurações de Usabilidade:**
    -   **Tema da Interface:** Escolha entre temas claro, escuro e **alto contraste** para melhor legibilidade e acessibilidade.
    -   **Tamanho da Fonte:** Ajuste o tamanho da fonte da interface.
    -   **Notificações Sonoras:** Sons para indicar eventos importantes como conclusão ou erro.
    -   **Notificações no Navegador:** Receba alertas do sistema operacional sobre o status da execução.
-   **Containerização Docker:** Backend e Frontend executam em contêineres Docker isolados, orquestrados por `docker-compose`, garantindo um ambiente de desenvolvimento e produção consistente.

## ⚙️ Tecnologias Utilizadas

-   **Backend:**
    -   Python 3.9
    -   FastAPI (API RESTful)
    -   PyTorch (Treinamento da CNN)
    -   Numba (Otimização de cálculos de métricas)
    -   Scikit-learn (Divisão de dados)
-   **Frontend:**
    -   HTML5, CSS3, JavaScript puro
    -   Chart.js (Geração de gráficos)
    -   Nginx (Servidor web estático e Proxy Reverso)
-   **Containerização:**
    -   Docker
    -   Docker Compose

## 🚀 Como Executar o Projeto

Certifique-se de ter o [Docker](https://www.docker.com/get-started/) e o [Docker Compose](https://docs.docker.com/compose/install/) instalados em sua máquina.

1.  **Clone o Repositório:**
    ```bash
    git clone [https://github.com/eliangomes33/PROJETO_AG_VSA.git](https://github.com/eliangomes33/PROJETO_AG_VSA.git)
    cd PROJETO_AG_VSA
    ```

2.  **Estrutura de Pastas e Arquivos (Verifique):**
    Confirme que a estrutura do seu projeto está correta. A pasta `data/` será criada automaticamente pelo Docker quando o dataset CIFAR-10 for baixado.
    ```
    .
    ├── backend/
│   ├── main.py
│   ├── genetic_algorithm.py
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── sobre.html
│   ├── algoritmo.html
│   ├── relatorios.html
│   ├── configuracoes.html
│   ├── ajuda.html
│   └── assets/
│       ├── css/
│       │   └── style.css
│       ├── js/
│       │   ├── script.js
│       │   └── theme.js
│       └── audio/ 
│           ├── success.mp3
│           └── error.mp3
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
└── nginx.conf
    ```
    *Para as notificações sonoras funcionarem, a pasta `frontend/assets/audio/` deve existir e conter os arquivos `success.mp3` e `error.mp3`.*

3.  **Construa e Inicie os Contêineres:**
    No diretório raiz do projeto, execute:
    ```bash
    docker compose up --build
    ```
    Este comando irá:
    -   Construir as imagens Docker para o backend e frontend (garantindo as últimas atualizações de código).
    -   Criar e iniciar os contêineres, configurando a rede interna e os mapeamentos de porta.
    -   O dataset CIFAR-10 será baixado automaticamente para a pasta `data/` na raiz do seu projeto (e persistirá entre as execuções). Esta pasta é ignorada pelo Git.

4.  **Acesse a Aplicação:**
    Abra seu navegador web e acesse:
    ```
    http://localhost:8001
    ```

## 📚 Manual de Uso (Perguntas Frequentes)

### 1. Como utilizar o aplicativo para otimizar a CNN?

Para executar o Algoritmo Genético e otimizar a CNN, siga estes passos:

1.  **Acessar a Página "Início"**: Certifique-se de estar na página principal (Home - `index.html`) onde o formulário de configuração do Algoritmo Genético está localizado.
2.  **Configurar os Parâmetros do Algoritmo Genético**: Preencha os campos do formulário com os hiperparâmetros desejados para o AG:
    * **Tamanho da População:** Define quantos conjuntos de hiperparâmetros (indivíduos) serão avaliados em cada geração.
    * **Número de Gerações:** Quantas "rodadas" o algoritmo genético irá executar.
    * **Taxa de Mutação:** A probabilidade de um hiperparâmetro ser alterado aleatoriamente.
    * **Épocas por Modelo:** O número de épocas que cada SmallCNN será treinada para ter sua acurácia (fitness) avaliada.
    * **Opções de Hiperparâmetros:** Defina as listas de valores possíveis para Taxas de Aprendizado, Tamanhos de Batch, Número de Filtros, Neurônios FC e Taxas de Dropout.
3.  **Iniciar a Execução**: Clique no botão "<strong style="color: var(--primary-color);">Executar AG</strong>". Um indicador de carregamento (spinner) aparecerá, e a seção "Progresso Atual" ficará visível.
4.  **Acompanhar o Progresso em Tempo Real**: Monitore o progresso na seção "Progresso Atual" (Geração atual, Indivíduo avaliado, Acurácia Global) e no "Log de Execução" (mensagens detalhadas). O gráfico de acurácias será atualizado dinamicamente.
5.  **Interromper a Execução (Opcional)**: Caso deseje parar o algoritmo antes que ele termine, clique no botão "<strong style="color: var(--stop-button-bg);">Parar AG</strong>". O algoritmo tentará parar de forma segura após finalizar a tarefa atual.
6.  **Visualizar os Resultados Finais**: Ao final da execução (ou após uma interrupção), a seção "Resultados Finais do Algoritmo Genético" será exibida com o relatório detalhado, gráfico e exemplos de imagens.
7.  **Iniciar Nova Execução**: Após a conclusão, o botão "<strong style="color: var(--primary-color);">Executar AG</strong>" será reativado, permitindo novas otimizações.

### 2. O que é um Algoritmo Genético (AG)?

Um Algoritmo Genético (AG) é uma técnica de otimização e busca inspirada na evolução biológica. Ele simula processos como seleção natural, crossover e mutação para encontrar soluções aproximadas para problemas complexos de otimização. No contexto deste projeto, o AG busca a melhor combinação de hiperparâmetros para uma rede neural.

### 3. O que são Redes Neurais Convolucionais (CNNs) e por que otimizar seus hiperparâmetros?

**Redes Neurais Convolucionais (CNNs)** são redes neurais especializadas para análise de dados visuais (imagens e vídeos). Elas são muito eficazes na detecção de padrões e características em imagens através de "camadas convolucionais".

**Hiperparâmetros** são configurações do modelo que são definidas *antes* do treinamento (ex: taxa de aprendizado, tamanho do batch, número de filtros). A escolha correta desses hiperparâmetros é vital para o desempenho da CNN. Otimizá-los é importante porque uma seleção manual é demorada e ineficiente, e pode levar a um modelo com baixa performance. O AG automatiza essa busca por combinações ideais.

### 4. O que é o dataset CIFAR-10?

O **CIFAR-10** é um conjunto de dados padrão para pesquisa em visão computacional. Ele consiste em 60.000 imagens coloridas de 32x32 pixels, divididas em 10 classes distintas (avião, automóvel, pássaro, gato, cervo, cachorro, sapo, cavalo, navio e caminhão). É amplamente utilizado para treinar e testar algoritmos de classificação de imagens, sendo o dataset que sua SmallCNN utiliza neste projeto.

### 5. O que significam as métricas 'Acurácia' e 'Precisão Média'?

* **Acurácia:** É a porcentagem de predições corretas que o modelo fez sobre o total de predições. É a métrica mais comum e indica o quão bem o modelo classificou as imagens em geral.
* **Precisão Média (Mean Precision):** A precisão, para cada classe, mede a proporção de predições positivas que foram realmente corretas (ex: de todas as vezes que o modelo disse que algo era um "avião", quantas vezes ele acertou?). A Precisão Média é a média dessas precisões calculadas para cada uma das 10 classes do dataset CIFAR-10, fornecendo uma visão mais balanceada do desempenho do modelo em todas as categorias.

### 6. Os resultados não estão bons. Como posso melhorar?

Se a acurácia final não está satisfatória, você pode tentar as seguintes estratégias:

* **Aumentar o Tamanho da População:** Mais indivíduos por geração permitem uma exploração mais ampla do espaço de hiperparâmetros.
* **Aumentar o Número de Gerações:** Mais gerações dão ao AG mais tempo para evoluir as soluções e convergir para um ótimo.
* **Aumentar as Épocas por Modelo (<code>ag_epochs</code>):** Permite que cada CNN individual treine por mais tempo, resultando em avaliações de fitness mais precisas e modelos potencialmente melhores. Este fator tem grande impacto no tempo de execução.
* **Expandir as Opções de Hiperparâmetros:** Forneça uma gama maior de valores nas listas (Taxas de Aprendizado, Tamanhos de Batch, etc.) para que o AG tenha mais combinações para explorar.
* **Ajustar a Taxa de Mutação:** Experimente taxas de mutação um pouco mais altas (para explorar mais) ou mais baixas (para refinar a busca).
* **Verificar Consistência:** Garanta que os valores fornecidos nas opções de hiperparâmetros são compatíveis com o modelo e o dataset (por exemplo, `batch_size` que seja um divisor do tamanho do dataset, `n_filters` em potências de 2, etc.).
* **Considerar Hardware:** Se a execução estiver muito lenta, considere usar uma GPU, pois ela acelera drasticamente o treinamento de CNNs.

## 🤝 Contribuição

Se você deseja contribuir para este projeto, sinta-se à vontade para fazer um fork do repositório e enviar Pull Requests.


---

## 📞 Contato

Para dúvidas ou sugestões, entre em contato com a equipe de desenvolvimento:

-   **Emails:**
    -   lianicolechaves@gmail.com
    -   Jossylynnovo123@gmail.com
    -   eliangomes3312@gmail.com
-   **Repositório:** https://github.com/eliangomes33/PROJETO_AG_VSA.git

---
