# backend/main.py
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import logging
import json
import time
from concurrent.futures import ThreadPoolExecutor
import functools

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Otimização de CNN com Algoritmo Genético",
    description="API RESTful para executar e monitorar um Algoritmo Genético que otimiza hiperparâmetros de CNNs para o dataset CIFAR-10.",
    version="1.0.0"
)

# Configuração CORS para permitir requisições do frontend
origins = [
    "http://localhost",
    "http://localhost:8001", # Porta do frontend
    "http://127.0.0.1",
    "http://127.0.0.1:8001", # Porta do frontend
    #
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pool de threads para executar o AG (CPU-bound) sem bloquear o loop de eventos da FastAPI
executor = ThreadPoolExecutor(max_workers=2) 

# Estado da execução do AG
class GAState:
    def __init__(self):
        self.is_running = False
        self.current_results = None
        self.websocket_connections: list[WebSocket] = []
        self.log_messages: list[str] = [] # Armazena logs para novos clientes
        self._stop_event = asyncio.Event() # Novo: Evento para sinalizar interrupção

    async def send_update(self, message: dict): # Agora espera um dicionário (serializável em JSON)
        # Serializa o dicionário para JSON antes de enviar via WebSocket
        json_message = json.dumps(message) 
        self.log_messages.append(json_message) # Armazena a string JSON
        
        for connection in self.websocket_connections:
            try:
                await connection.send_text(json_message) # Envia como texto JSON
            except RuntimeError as e:
                logger.warning(f"Erro ao enviar mensagem WebSocket: {e}")
                # A conexão pode ter sido fechada, será removida na próxima verificação


ga_state = GAState()

# Modelo de requisição para o endpoint /run
class GAParams(BaseModel):
    pop_size: int = 6
    generations: int = 10
    mutation_rate: float = 0.3
    ag_epochs: int = 5
    learning_rate_options: list[float] = [0.01, 0.005, 0.001, 0.0005]
    batch_size_options: list[int] = [8, 16, 32]
    n_filters_options: list[int] = [16, 32, 64]
    n_fc_options: list[int] = [64, 128, 256]
    dropout_options: list[float] = [0.0, 0.25, 0.5]


@app.post("/run-genetic-algorithm", summary="Inicia a execução do Algoritmo Genético")
async def run_ga(params: GAParams):
    if ga_state.is_running:
        raise HTTPException(status_code=400, detail="Algoritmo Genético já está em execução.")

    ga_state.is_running = True
    ga_state.current_results = None
    ga_state.log_messages = [] # Limpa logs anteriores
    ga_state._stop_event.clear() # Limpa qualquer sinal de interrupção anterior

    main_loop = asyncio.get_running_loop()

    def sync_progress_callback(message: dict): # Agora espera um dicionário
        logger.info(f"GA Progress (sync callback): {message.get('message', str(message))}")
        asyncio.run_coroutine_threadsafe(ga_state.send_update(message), main_loop)

    def _run_ga_blocking(params_dict, progress_cb_sync, stop_event):
        from genetic_algorithm import run_genetic_algorithm 

        return run_genetic_algorithm(
            pop_size=params_dict["pop_size"],
            generations=params_dict["generations"],
            mutation_rate=params_dict["mutation_rate"],
            ag_epochs=params_dict["ag_epochs"],
            learning_rate_options=params_dict["learning_rate_options"],
            batch_size_options=params_dict["batch_size_options"],
            n_filters_options=params_dict["n_filters_options"],
            n_fc_options=params_dict["n_fc_options"],
            dropout_options=params_dict["dropout_options"],
            progress_callback=progress_cb_sync,
            stop_event_flag=stop_event # Passa o evento de parada
        )

    async def run_task():
        try:
            logger.info(f"Iniciando AG com parâmetros: {params.dict()}")
            
            results = await main_loop.run_in_executor(
                executor,
                functools.partial(_run_ga_blocking, params.dict(), sync_progress_callback, ga_state._stop_event)
            )
            
            ga_state.current_results = results
            logger.info("Algoritmo Genético concluído e resultados armazenados.")
        except Exception as e:
            logger.error(f"Erro durante a execução do AG: {e}", exc_info=True)
            await ga_state.send_update({"type": "error", "message": f"ERRO: A execução do AG falhou: {str(e)}"})
        finally:
            ga_state.is_running = False
            ga_state._stop_event.clear() # Limpa o evento de parada ao final, caso não tenha sido setado

    asyncio.create_task(run_task())

    return {"message": "Algoritmo Genético iniciado com sucesso!", "status": "running"}

@app.post("/stop-genetic-algorithm", summary="Interrompe a execução do Algoritmo Genético")
async def stop_ga():
    if not ga_state.is_running:
        return {"message": "Algoritmo Genético não está em execução.", "status": "not_running"}
    
    ga_state._stop_event.set() # Seta o evento para sinalizar a interrupção
    logger.info("Sinal de interrupção para o AG enviado.")
    await ga_state.send_update({"type": "info", "message": "Sinal de interrupção enviado ao Algoritmo Genético. Aguardando finalização da tarefa atual..."})
    return {"message": "Sinal de interrupção enviado ao Algoritmo Genético.", "status": "stopping"}


@app.get("/status", summary="Retorna o status atual da execução do AG")
async def get_status():
    return {"is_running": ga_state.is_running}

@app.get("/results", summary="Retorna os resultados finais do AG")
async def get_results():
    if ga_state.is_running:
        return {"message": "Algoritmo Genético ainda está em execução. Por favor, aguarde."}
    if ga_state.current_results:
        return ga_state.current_results
    return {"message": "Nenhum resultado disponível. O AG ainda não foi executado ou falhou."}


@app.websocket("/ws/log")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    ga_state.websocket_connections.append(websocket)
    logger.info(f"Novo cliente WebSocket conectado: {websocket.client.host}:{websocket.client.port}")

    for msg_str in ga_state.log_messages:
        try:
            await websocket.send_text(msg_str)
        except RuntimeError:
            logger.warning("Conexão WebSocket fechada antes de enviar o histórico.")
            break

    try:
        while True:
            data = await websocket.receive_text()
            logger.debug(f"Mensagem WebSocket recebida: {data}")
    except WebSocketDisconnect:
        ga_state.websocket_connections.remove(websocket)
        logger.info(f"Cliente WebSocket desconectado: {websocket.client.host}:{websocket.client.port}")
    except Exception as e:
        logger.error(f"Erro no WebSocket: {e}", exc_info=True)
        ga_state.websocket_connections.remove(websocket)