// app/frontend/script.js

document.addEventListener('DOMContentLoaded', () => {
    // --- Lógica do Menu Toggle (centralizada para todas as páginas) ---
    const menuToggle = document.createElement('button');
    menuToggle.className = 'menu-toggle';
    document.body.appendChild(menuToggle);
    menuToggle.textContent = 'Menu'; 
    
    const sidebar = document.querySelector('.sidebar');
    menuToggle.addEventListener('click', () => {
        sidebar.classList.toggle('active');
    });

    // Fechar menu ao clicar em um link (mobile)
    document.querySelectorAll('.sidebar a').forEach(link => {
        link.addEventListener('click', () => {
            if (window.innerWidth <= 768) {
                sidebar.classList.remove('active');
            }
        });
    });
    // --- Fim da Lógica do Menu Toggle ---

    const backendUrl = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
        ? 'http://localhost:8000' 
        : '/api'; 

    const backendWsUrl = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
        ? 'ws://localhost:8000/ws/log' 
        : `ws://${window.location.host}/api/ws/log`; 

    // Se estivermos na página principal (index.html)
    if (document.getElementById('ga-form')) {
        const form = document.getElementById('ga-form');
        const runButton = document.getElementById('run-button');
        const stopButton = document.getElementById('stop-button');
        const exportResultsButton = document.getElementById('export-results-btn');

        const loadingSpinner = document.getElementById('loading-spinner');
        const resultsContainer = document.getElementById('results-container');
        const errorMessage = document.getElementById('error-message');
        const executionLog = document.getElementById('execution-log');
        const statusMessageSpan = document.getElementById('status-message');

        // Elementos para progresso em tempo real
        const currentProgressContainer = document.getElementById('current-progress');
        const currentGenerationSpan = document.getElementById('current-generation');
        const totalGenerationsSpan = document.getElementById('total-generations');
        const currentIndividualSpan = document.getElementById('current-individual');
        const currentPopSizeSpan = document.getElementById('current-pop-size');
        const individualAccuracySpan = document.getElementById('individual-accuracy');
        const bestGlobalIndividualSummarySpan = document.getElementById('best-global-individual-summary');
        // CORREÇÃO: Erro de digitação 'document = ' removido aqui.
        const bestGlobalAccuracySummarySpan = document.getElementById('best-global-accuracy-summary'); 

        // Elementos para o melhor de cada geração
        const bestOfEachGenerationContainer = document.getElementById('best-of-each-generation-container'); 
        const generationBestListDiv = document.getElementById('generation-best-list');

        // Elementos para resultados finais
        const finalTotalTimeSpan = document.getElementById('final-total-time');
        const finalBestIndividualSpan = document.getElementById('final-best-individual');
        const finalBestAccuracySpan = document.getElementById('final-best-accuracy');
        const finalMeanPrecisionSpan = document.getElementById('final-mean-precision');
        const interruptionMessageSpan = document.getElementById('interruption-message');

        // Elementos para imagens de acertos/erros
        const imageExamplesContainer = document.getElementById('image-examples-container');
        const correctPredictionsDiv = document.getElementById('correct-predictions');
        const incorrectPredictionsDiv = document.getElementById('incorrect-predictions');

        const accuracyChartCanvas = document.getElementById('accuracyChart');
        let accuracyChart = null;

        let logWebSocket = null;
        let lastGAresults = null;

        // Função para gerenciar o estado dos botões
        function setButtonState(running) {
            runButton.disabled = running;
            stopButton.style.display = running ? 'inline-block' : 'none';
            exportResultsButton.disabled = running;
        }

        function updateLog(message) {
            const logEntry = document.createElement('div');
            const displayMessage = typeof message === 'object' && message !== null && message.message 
                                 ? message.message 
                                 : String(message);
            
            logEntry.textContent = `${new Date().toLocaleTimeString()}: ${displayMessage}`;
            executionLog.appendChild(logEntry);
            executionLog.scrollTop = executionLog.scrollHeight;
            
            if (typeof message === 'object' && message !== null) {
                if (message.type === 'error') {
                    playSound('error');
                    showNotification("Erro na Aplicação", "Um erro inesperado ocorreu. Verifique o log.");
                } else if (message.type === 'final_results') {
                    playSound('success');
                    showNotification("Execução Concluída", "O Algoritmo Genético finalizou a execução!");
                } else if (message.type === 'new_best_global') {
                    // Opcional: som diferente para novo melhor global
                }
            }
        }

        function connectWebSocket() {
            if (logWebSocket) {
                logWebSocket.close();
            }
            logWebSocket = new WebSocket(backendWsUrl);

            logWebSocket.onopen = (event) => {
                updateLog({type: "info", message: "Conectado ao log do backend (WebSocket)."});
                console.log('WebSocket Open:', event);
            };

            logWebSocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    handleWebSocketMessage(data);
                } catch (e) {
                    updateLog(event.data);
                }
            };

            logWebSocket.onclose = (event) => {
                updateLog({type: "info", message: "Desconectado do log do backend (WebSocket). Tentando reconectar em 5s..."});
                console.log('WebSocket Closed:', event);
                setTimeout(connectWebSocket, 5000);
            };

            logWebSocket.onerror = (error) => {
                updateLog({type: "error", message: "Erro no WebSocket. Verifique o backend e o console."});
                console.error('WebSocket Error:', error);
            };
        }

        function handleWebSocketMessage(data) {
            updateLog(data);
            switch (data.type) {
                case "init":
                    break;
                case "generation_start":
                    currentProgressContainer.style.display = 'block';
                    currentGenerationSpan.textContent = data.generation;
                    totalGenerationsSpan.textContent = data.total_generations;
                    currentIndividualSpan.textContent = '0';
                    individualAccuracySpan.textContent = 'N/A';
                    statusMessageSpan.textContent = `Iniciando Geração ${data.generation}/${data.total_generations}...`;
                    break;
                case "individual_eval":
                    currentIndividualSpan.textContent = data.individual_idx;
                    currentPopSizeSpan.textContent = data.pop_size;
                    individualAccuracySpan.textContent = 'Aguardando...'; 
                    statusMessageSpan.textContent = `Avaliando Indivíduo ${data.individual_idx}/${data.pop_size} (Geração ${data.generation})...`;
                    break;
                case "new_best_global":
                    bestGlobalIndividualSummarySpan.textContent = JSON.stringify(data.best_individual, null, 2);
                    bestGlobalAccuracySummarySpan.textContent = (data.best_accuracy * 100).toFixed(2) + '%';
                    statusMessageSpan.textContent = `Novo Melhor Global encontrado na Geração ${data.current_generation}!`;
                    
                    updateAccuracyChart(data.history_accuracies); 
                    break;
                case "generation_end":
                    statusMessageSpan.textContent = `Geração ${data.generation} Concluída. Melhor da Geração: Acurácia ${(data.best_accuracy_gen * 100).toFixed(2)}%`;
                    
                    addGenerationBestEntry({
                        generation: data.generation,
                        accuracy: data.best_accuracy_gen,
                        precision: data.best_precision_gen,
                        individual: data.best_individual_gen,
                        eval_time: data.eval_time_gen,
                        total_time_this_gen: data.total_time_this_gen,
                        history_accuracies: data.history_accuracies
                    });
                    
                    updateAccuracyChart(data.history_accuracies);
                    break;
                case "final_results":
                    displayResults(data.data);
                    currentProgressContainer.style.display = 'none';
                    statusMessageSpan.textContent = "Algoritmo Genético Concluído!";
                    setButtonState(false);
                    break;
                case "info":
                    statusMessageSpan.textContent = data.message;
                    break;
                case "warning":
                    statusMessageSpan.textContent = data.message;
                    break;
                case "error":
                    displayError(data.message);
                    statusMessageSpan.textContent = `ERRO: ${data.message}`;
                    setButtonState(false);
                    break;
                default:
                    console.log("Mensagem WebSocket desconhecida:", data);
                    break;
            }
        }

        function addGenerationBestEntry(genData) {
            bestOfEachGenerationContainer.style.display = 'block';
            const entryDiv = document.createElement('div');
            entryDiv.className = 'generation-best-entry';
            entryDiv.innerHTML = `
                <h4>Geração ${genData.generation}:</h4>
                <p><strong>Acurácia:</strong> ${(genData.accuracy * 100).toFixed(2)}%</p>
                <p><strong>Precisão:</strong> ${(genData.precision * 100).toFixed(2)}%</p>
                <p><strong>Tempo de Avaliação (melhor indivíduo):</strong> ${genData.eval_time.toFixed(2)}s</p>
                <p><strong>Tempo Total da Geração:</strong> ${genData.total_time_this_gen.toFixed(2)}s</p>
                <p><strong>Hiperparâmetros:</strong> <code>${JSON.stringify(genData.individual, null, 2)}</code></p>
            `;
            generationBestListDiv.appendChild(entryDiv);
            generationBestListDiv.scrollTop = generationBestListDiv.scrollHeight;
        }

        function updateAccuracyChart(history_accuracies) {
            const generations = history_accuracies.length;
            const labels = Array.from({ length: generations }, (_, i) => `Geração ${i + 1}`); 
            const datasets = [];

            if (history_accuracies.length > 0) {
                const numTopIndividuals = history_accuracies[0].length;

                for (let i = 0; i < numTopIndividuals; i++) {
                    const accuracies = history_accuracies.map(gen => gen[i] * 100);
                    datasets.push({
                        label: `Melhor Indivíduo ${i + 1}`,
                        data: accuracies,
                        borderColor: `hsl(${i * 60}, 70%, 50%)`,
                        fill: false,
                        tension: 0.1
                    });
                }
            }

            if (!accuracyChartCanvas) {
                console.error("Elemento canvas com ID 'accuracyChart' não encontrado. Não é possível renderizar o gráfico.");
                return;
            }

            if (accuracyChart) {
                accuracyChart.data.labels = labels;
                accuracyChart.data.datasets = datasets;
                accuracyChart.update();
            } else {
                accuracyChart = new Chart(accuracyChartCanvas, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: datasets
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Evolução da Acurácia dos Melhores Indivíduos'
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: {
                                    display: true,
                                    text: 'Acurácia (%)'
                                },
                                max: 100 
                            },
                            x: {
                                title: {
                                    display: true,
                                    text: 'Geração'
                                }
                            }
                        }
                    }
                });
            }
        }

        connectWebSocket();

        stopButton.addEventListener('click', async () => {
            updateLog({type: "info", message: "Enviando sinal de interrupção para o AG..."});
            stopButton.disabled = true;

            try {
                const response = await fetch(`${backendUrl}/stop-genetic-algorithm`, {
                    method: 'POST',
                });
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || `Erro HTTP: ${response.status}`);
                }
                const data = await response.json();
                updateLog({type: "info", message: data.message});
            } catch (error) {
                console.error('Erro ao enviar sinal de parada:', error);
                displayError(`Falha ao enviar sinal de parada: ${error.message}`);
            } finally {
                stopButton.disabled = false;
            }
        });

        exportResultsButton.addEventListener('click', () => {
            if (lastGAresults) {
                const filename = "resultados_ag.json";
                const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(lastGAresults, null, 2));
                const downloadAnchorNode = document.createElement('a');
                downloadAnchorNode.setAttribute("href", dataStr);
                downloadAnchorNode.setAttribute("download", filename);
                document.body.appendChild(downloadAnchorNode);
                downloadAnchorNode.click();
                downloadAnchorNode.remove();
                updateLog({type: "info", message: `Resultados exportados para ${filename}`});
            } else {
                updateLog({type: "warning", message: "Nenhum resultado para exportar. Execute o AG primeiro."});
            }
        });


        form.addEventListener('submit', async (event) => {
            event.preventDefault();

            // Limpa resultados e mensagens anteriores
            resultsContainer.style.display = 'none';
            errorMessage.style.display = 'none';
            executionLog.innerHTML = '';
            imageExamplesContainer.style.display = 'none';
            correctPredictionsDiv.innerHTML = '';
            incorrectPredictionsDiv.innerHTML = '';
            interruptionMessageSpan.style.display = 'none';
            
            // Limpa a seção "Melhores da Geração"
            bestOfEachGenerationContainer.style.display = 'none';
            generationBestListDiv.innerHTML = '';

            currentProgressContainer.style.display = 'none';
            currentGenerationSpan.textContent = '0';
            totalGenerationsSpan.textContent = '0';
            currentIndividualSpan.textContent = '0';
            currentPopSizeSpan.textContent = '0';
            individualAccuracySpan.textContent = 'N/A';
            bestGlobalIndividualSummarySpan.textContent = 'N/A';
            bestGlobalAccuracySummarySpan.textContent = 'N/A';


            if (accuracyChart) {
                accuracyChart.destroy();
                accuracyChart = null;
            }
            updateAccuracyChart([]);

            setButtonState(true);
            loadingSpinner.style.display = 'block';
            statusMessageSpan.textContent = "Verificando status do AG...";


            try {
                const statusResponse = await fetch(`${backendUrl}/status`);
                if (!statusResponse.ok) {
                    throw new Error(`Erro ao verificar status: ${statusResponse.status}`);
                }
                const statusData = await statusResponse.json();
                if (statusData.is_running) {
                    displayError("Algoritmo Genético já está em execução. Aguarde a conclusão ou tente novamente mais tarde.");
                    setButtonState(true);
                    loadingSpinner.style.display = 'none';
                    return;
                }
            } catch (error) {
                displayError(`Falha ao verificar status do backend: ${error.message}`);
                setButtonState(false);
                loadingSpinner.style.display = 'none';
                return;
            }

            const pop_size = parseInt(document.getElementById('pop_size').value);
            const generations = parseInt(document.getElementById('generations').value);
            const mutation_rate = parseFloat(document.getElementById('mutation_rate').value);
            const ag_epochs = parseInt(document.getElementById('ag_epochs').value);

            const parseNumberList = (id, type = 'float') => {
                const value = document.getElementById(id).value;
                return value.split(',').map(s => {
                    const num = type === 'float' ? parseFloat(s.trim()) : parseInt(s.trim());
                    if (isNaN(num)) throw new Error(`Erro: Valor inválido em ${id}: '${s.trim()}' não é um número válido.`);
                    return num;
                }).filter(n => !isNaN(n));
            };

            let params;
            try {
                params = {
                    pop_size,
                    generations,
                    mutation_rate,
                    ag_epochs,
                    learning_rate_options: parseNumberList('learning_rate_options', 'float'),
                    batch_size_options: parseNumberList('batch_size_options', 'int'),
                    n_filters_options: parseNumberList('n_filters_options', 'int'),
                    n_fc_options: parseNumberList('n_fc_options', 'int'),
                    dropout_options: parseNumberList('dropout_options', 'float')
                };
                updateLog({type: "info", message: "Parâmetros coletados do formulário."});
                console.log("Parâmetros enviados:", params);
            } catch (error) {
                displayError(error.message);
                setButtonState(false);
                loadingSpinner.style.display = 'none';
                return;
            }

            try {
                updateLog({type: "info", message: "Enviando requisição para o backend..."});
                updateLog({type: "info", message: "Iniciando algoritmo genético via backend..."});
                const response = await fetch(`${backendUrl}/run-genetic-algorithm`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(params),
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || `Erro HTTP: ${response.status}`);
                }

                const data = await response.json();
                updateLog({type: "info", message: data.message || "Requisição enviada com sucesso."});
                updateLog({type: "info", message: "Execução em progresso. Acompanhe o log e o progresso acima..."});

                const checkResultsInterval = setInterval(async () => {
                    const statusRes = await fetch(`${backendUrl}/status`);
                    const statusDat = await statusRes.json();

                    if (!statusDat.is_running) {
                        clearInterval(checkResultsInterval);
                        updateLog({type: "info", message: "AG terminou. Buscando resultados finais..."});
                        const finalResultsRes = await fetch(`${backendUrl}/results`);
                        if (finalResultsRes.ok) {
                            const finalData = await finalResultsRes.json();
                            lastGAresults = finalData;
                            console.log("Dados recebidos (via polling):", finalData);
                            displayResults(finalData);
                        } else {
                            displayError(`Erro ao buscar resultados finais: ${finalResultsRes.status}`);
                        }
                        setButtonState(false);
                        loadingSpinner.style.display = 'none';
                        currentProgressContainer.style.display = 'none';
                    }
                }, 5000);

            } catch (error) {
                console.error('Erro ao executar o AG:', error);
                displayError(`Falha na comunicação com o backend ou erro no AG: ${error.message}`);
                setButtonState(false);
                loadingSpinner.style.display = 'none';
            }
        });

        function displayResults(data) {
            resultsContainer.style.display = 'block';
            
            if (data.interrupted) {
                interruptionMessageSpan.style.display = 'block';
                updateLog({type: "info", message: "A execução do Algoritmo Genético foi interrompida pelo usuário."});
            } else {
                interruptionMessageSpan.style.display = 'none';
            }

            finalTotalTimeSpan.textContent = data.tempo_total_segundos.toFixed(2);
            finalBestIndividualSpan.textContent = JSON.stringify(data.melhor_individuo, null, 2);
            finalBestAccuracySpan.textContent = (data.melhor_acuracia * 100).toFixed(2) + '%';
            finalMeanPrecisionSpan.textContent = (data.melhor_precisao * 100).toFixed(2) + '%';

            updateAccuracyChart(data.historico_acuracias);

            if (data.correct_examples && data.correct_examples.length > 0 ||
                data.incorrect_examples && data.incorrect_examples.length > 0) {
                imageExamplesContainer.style.display = 'block';
                displayImageExamples(correctPredictionsDiv, data.correct_examples);
                displayImageExamples(incorrectPredictionsDiv, data.incorrect_examples);
            } else {
                updateLog({type: "warning", message: "Não há exemplos de imagens para exibir (pode ter sido interrompido ou dados insuficientes)."});
                imageExamplesContainer.style.display = 'none';
            }
        }

        function displayImageExamples(containerElement, images) {
            containerElement.innerHTML = '';
            if (!images || images.length === 0) {
                containerElement.innerHTML = '<p>Nenhum exemplo para exibir.</p>';
                return;
            }
            images.forEach(imgData => {
                const imgCard = document.createElement('div');
                imgCard.className = 'image-card';

                const img = document.createElement('img');
                img.src = imgData.image;
                img.alt = `Pred: ${imgData.predicted} / True: ${imgData.true_label}`;

                const imgTitle = document.createElement('div');
                imgTitle.className = 'image-title';
                imgTitle.innerHTML = `Pred: <strong>${imgData.predicted}</strong><br>True: ${imgData.true_label}`;

                imgCard.appendChild(img);
                imgCard.appendChild(imgTitle);
                containerElement.appendChild(imgCard);
            });
        }

        function displayError(message) {
            errorMessage.textContent = message;
            errorMessage.style.display = 'block';
            updateLog({type: "error", message: `ERRO: ${message}`});
        }
    }
});