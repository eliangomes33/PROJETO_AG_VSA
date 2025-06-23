# backend/genetic_algorithm.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import random
import numpy as np
import time
from numba import jit
import io
import base64
from PIL import Image
import asyncio # Importar asyncio para lidar com o Event

# Garante que o benchmark do cuDNN esteja ativado para melhor desempenho em GPUs
torch.backends.cudnn.benchmark = True

# Definição do espaço de hiperparâmetros (pode ser ajustado dinamicamente)
DEFAULT_HYPERPARAMETER_SPACE = {
    "learning_rate": [1e-2, 5e-3, 5e-1, 1e-3],
    "batch_size": [16, 32, 64],
    "n_filters": [64, 128, 256],
    "n_fc": [128, 256],
    "dropout": [0.0, 0.01, 0.25],
    "activation": ['relu', 'leakyrelu'],
    "optimizer": ["adam", "sgd"]
}

# Carregamento de Dados CIFAR-10 (Adaptado para 32x32)
def load_cifar10_data(train_subset_size=30000, val_subset_size=10000):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Redimensiona para 32x32 para CIFAR-10
        transforms.ToTensor(),
        # Normalização para CIFAR-10
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

    # Usa subconjuntos para agilizar o treinamento durante a otimização com AG
    train_subset = Subset(dataset, train_idx[:train_subset_size])
    val_subset = Subset(dataset, val_idx[:val_subset_size])
    full_val_set = Subset(dataset, val_idx) # Para avaliação final, se necessário

    return train_subset, val_subset, full_val_set

# Modelo CNN
class SmallCNN(nn.Module):
    def __init__(self, n_filters, n_fc, dropout, activation):
        super().__init__()
        act_dict = {
            'relu': nn.ReLU(),
            'leakyrelu': nn.LeakyReLU()
        }
        self.activation_fn = act_dict[activation]
        self.conv1 = nn.Conv2d(3, n_filters, 3, padding=1)
        self.conv2 = nn.Conv2d(n_filters, n_filters * 2, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = None
        self.fc2 = None
        self.n_fc = n_fc
        self.num_classes = 10  # CIFAR-10 tem 10 classes

    def build(self, device):
        # Calcula flat_dim dinamicamente
        with torch.no_grad():
            x = torch.zeros(1, 3, 32, 32).to(device)
            x = self.pool(self.activation_fn(self.conv1(x)))
            x = self.pool(self.activation_fn(self.conv2(x)))
            flat_dim = x.view(1, -1).shape[1]
        self.fc1 = nn.Linear(flat_dim, self.n_fc).to(device)
        self.fc2 = nn.Linear(self.n_fc, self.num_classes).to(device)

    def forward(self, x):
        x = self.pool(self.activation_fn(self.conv1(x)))
        x = self.pool(self.activation_fn(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.activation_fn(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# Função JIT para cálculo de métricas (otimizada com Numba)
@jit(nopython=True)
def calculate_metrics(preds, labels):
    correct_predictions = 0
    num_classes = 10
    true_positives_per_class = np.zeros(num_classes, dtype=np.float64)
    predicted_positives_per_class = np.zeros(num_classes, dtype=np.float64)

    preds = preds.astype(np.int64)
    labels = labels.astype(np.int64)

    for i in range(len(preds)):
        if 0 <= labels[i] < num_classes:
            predicted_positives_per_class[preds[i]] += 1

        if 0 <= preds[i] < num_classes and 0 <= labels[i] < num_classes:
            if preds[i] == labels[i]:
                correct_predictions += 1
                true_positives_per_class[preds[i]] += 1

    precision_per_class = np.zeros(num_classes, dtype=np.float64)
    for i in range(num_classes):
        if predicted_positives_per_class[i] > 0:
            precision_per_class[i] = true_positives_per_class[i] / predicted_positives_per_class[i]
        else:
            precision_per_class[i] = 0.0

    accuracy = correct_predictions / len(preds) if len(preds) > 0 else 0.0
    mean_precision = np.mean(precision_per_class)

    return accuracy, mean_precision


# Função para avaliar o modelo
# Adicionado stop_event para permitir interrupção durante o treinamento de um modelo
def evaluate_model(individual_params, device, train_data, val_data, epochs=3, progress_callback=None, stop_event=None):
    train_loader = DataLoader(train_data, batch_size=individual_params["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=individual_params["batch_size"], shuffle=False, num_workers=2)

    model = SmallCNN(individual_params["n_filters"], individual_params["n_fc"],
                     individual_params["dropout"], individual_params["activation"]).to(device)
    model.build(device) # Construa o modelo após movê-lo para o device
    criterion = nn.CrossEntropyLoss()

    weight_decay = 1e-4

    if individual_params["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=individual_params["learning_rate"], weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=individual_params["learning_rate"], momentum=0.9, weight_decay=weight_decay)

    effective_scheduler_step_size = max(1, epochs // 3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=effective_scheduler_step_size, gamma=0.5)

    model.train()
    for epoch_idx in range(epochs):
        # Verifica a flag de interrupção em cada época
        if stop_event and stop_event.is_set():
            if progress_callback:
                progress_callback({"type": "info", "message": f"  Interrupção detectada na época {epoch_idx + 1}/{epochs}. Parando treinamento do indivíduo."})
            return 0.0, 0.0, np.array([]), np.array([]) # Retorna 0s ou valores que indiquem falha/interrupção

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if progress_callback:
            progress_callback({"type": "info", "message": f"  Epoch {epoch_idx + 1}/{epochs} for current individual."})


    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            _, predicted = torch.max(out, 1)
            preds.extend(predicted.cpu().numpy())
            labels.extend(yb.cpu().numpy())

    preds_np = np.array(preds)
    labels_np = np.array(labels)

    acc, precision = calculate_metrics(preds_np, labels_np)

    return acc, precision, preds_np, labels_np

# Função para obter exemplos de imagem em base64 para o frontend
def get_image_examples_for_frontend(dataset, preds, labels, acertos=True, n=5):
    if preds is None or labels is None or len(preds) == 0 or len(labels) == 0:
        return []

    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    if not isinstance(preds, np.ndarray):
        preds = np.array(preds)

    condition_indices = np.where((preds == labels) if acertos else (preds != labels))[0]
    
    # Seleciona aleatoriamente n índices, se houver mais de n
    if len(condition_indices) > n:
        idxs_to_plot = random.sample(list(condition_indices), n)
    else:
        idxs_to_plot = list(condition_indices) # Converte para lista

    # CIFAR-10 class names
    classes_cifar10 = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    image_data = []
    for idx_in_preds_labels in idxs_to_plot:
        if idx_in_preds_labels >= len(dataset):
            continue # Garante que o índice não exceda o tamanho do dataset

        img_tensor, true_label_idx_original = dataset[idx_in_preds_labels] # Obtenha o índice da label do dataset original
        
        # Desnormalização para plotting (usando CIFAR-10 stats)
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=img_tensor.device).view(3, 1, 1)
        std = torch.tensor([0.2471, 0.2435, 0.2616], device=img_tensor.device).view(3, 1, 1)
        
        img_tensor = img_tensor * std + mean
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        img_pil = Image.fromarray((img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        
        buffered = io.BytesIO()
        img_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        pred_idx = preds[idx_in_preds_labels]
        true_label_idx = labels[idx_in_preds_labels] # Use o rótulo do array 'labels' (que pode ser do subset)

        pred_class_name = classes_cifar10[pred_idx] if 0 <= pred_idx < len(classes_cifar10) else f"Class {pred_idx}"
        true_class_name = classes_cifar10[true_label_idx] if 0 <= true_label_idx < len(classes_cifar10) else f"Class {true_label_idx}"

        image_data.append({
            "image": f"data:image/png;base64,{img_str}",
            "predicted": pred_class_name,
            "true_label": true_class_name
        })
    return image_data


# Algoritmo Genético
# Adicionado stop_event_flag para permitir interrupção externa
def run_genetic_algorithm(
    pop_size: int,
    generations: int,
    mutation_rate: float,
    ag_epochs: int,
    learning_rate_options: list[float],
    batch_size_options: list[int],
    n_filters_options: list[int],
    n_fc_options: list[int],
    dropout_options: list[float],
    progress_callback=None, # Callback para reportar progresso
    stop_event_flag=None # Novo: Flag de evento para interrupção
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if progress_callback:
        progress_callback({"type": "init", "message": f"Dispositivo selecionado para treinamento: {device}"})

    # Atualiza o espaço de busca com as opções fornecidas pelo frontend
    current_space = {
        "learning_rate": learning_rate_options,
        "batch_size": batch_size_options,
        "n_filters": n_filters_options,
        "n_fc": n_fc_options,
        "dropout": dropout_options,
        "activation": ['relu', 'leakyrelu'], # Mantidos fixos por simplicidade no frontend atual
        "optimizer": ["adam", "sgd"] # Mantidos fixos
    }

    def create_individual():
        return {k: random.choice(v) for k, v in current_space.items()}

    # Crossover adaptado para aceitar 3 pais
    def crossover(p1, p2, p3):
        child = {}
        for k in current_space:
            child[k] = random.choice([p1[k], p2[k], p3[k]])
        return child

    def mutate(ind):
        k = random.choice(list(current_space.keys()))
        ind[k] = random.choice(current_space[k])
        return ind

    # Carrega os dados uma única vez
    trainset, valset, full_val_set = load_cifar10_data() # Carrega o full_val_set para exemplos visuais
    if progress_callback:
        progress_callback({"type": "init", "message": "Dados CIFAR-10 carregados e divididos."})

    population = [create_individual() for _ in range(pop_size)]
    history_accuracies = []
    best_individual = None
    best_accuracy = -1
    best_precision = -1
    best_preds_final = None # Predições do melhor modelo global
    best_labels_final = None # Labels do melhor modelo global
    start_time = time.time()
    
    interrupted = False

    for generation in range(generations):
        # Verifica a flag de interrupção antes de iniciar uma nova geração
        if stop_event_flag and stop_event_flag.is_set():
            interrupted = True
            if progress_callback:
                progress_callback({"type": "info", "message": f"Interrupção do AG detectada. Parando na Geração {generation + 1}."})
            break # Sai do loop de gerações

        if progress_callback:
            progress_callback({
                "type": "generation_start",
                "generation": generation + 1,
                "total_generations": generations
            })

        evaluations = []
        for i, ind in enumerate(population):
            # Verifica a flag de interrupção antes de avaliar cada indivíduo
            if stop_event_flag and stop_event_flag.is_set():
                interrupted = True
                if progress_callback:
                    progress_callback({"type": "info", "message": f"Interrupção do AG detectada. Parando antes de avaliar o Indivíduo {i+1} da Geração {generation + 1}."})
                break # Sai do loop de indivíduos

            if progress_callback:
                progress_callback({
                    "type": "individual_eval",
                    "generation": generation + 1,
                    "individual_idx": i + 1,
                    "pop_size": pop_size,
                    "hyperparameters": ind
                })
            
            acc, prec, preds, labels = evaluate_model(ind, device, train_data=trainset, val_data=valset, epochs=ag_epochs, progress_callback=progress_callback, stop_event=stop_event_flag)
            
            # Se a avaliação do modelo foi interrompida, sai do loop de indivíduos
            if stop_event_flag and stop_event_flag.is_set():
                interrupted = True
                break
            
            evaluations.append((ind, acc, prec, preds, labels))
        
        # Se o loop de indivíduos foi interrompido, interrompe também o loop de gerações
        if interrupted:
            break

        evaluations.sort(key=lambda x: x[1], reverse=True)

        # Guarda as 4 melhores acurácias da geração (ou menos, se a população for menor)
        history_accuracies.append([e[1] for e in evaluations[:min(len(evaluations), 4)]])

        current_gen_best_acc = evaluations[0][1]
        current_gen_best_ind = evaluations[0][0]

        if current_gen_best_acc > best_accuracy:
            best_individual, best_accuracy, best_precision, best_preds_final, best_labels_final = evaluations[0]
            if progress_callback:
                progress_callback({
                    "type": "new_best_global",
                    "best_accuracy": best_accuracy,
                    "best_individual": best_individual,
                    "current_generation": generation + 1,
                    "history_accuracies": history_accuracies # Envia o histórico completo para atualização do gráfico
                })

        if progress_callback:
            progress_callback({
                "type": "generation_end",
                "generation": generation + 1,
                "best_accuracy_gen": current_gen_best_acc,
                "best_individual_gen": current_gen_best_ind
            })

        new_population = []
        # Elitismo: Mantém os 2 melhores indivíduos para a próxima geração
        if len(evaluations) >= 2:
            new_population.extend([evaluations[0][0], evaluations[1][0]])
        elif len(evaluations) == 1: # Se só um foi avaliado, mantém ele
            new_population.extend([evaluations[0][0]])
        # else: Se nenhuma avaliação foi bem-sucedida, new_population fica vazia, e o loop abaixo criará pop_size indivíduos aleatórios.

        while len(new_population) < pop_size:
            # Seleção de 3 pais para crossover, priorizando os melhores
            parents_pool = [e[0] for e in evaluations[:min(len(evaluations), max(2, pop_size // 2))]] # Pega os top X para serem pais
            
            # Garante que haja pais para a seleção, mesmo que repetidos para pop_size pequenos
            if len(parents_pool) < 3 and len(evaluations) > 0:
                # Se não há 3 pais distintos entre os melhores, duplica os disponíveis
                available_parents = [e[0] for e in evaluations] # Todos os avaliados
                if len(available_parents) >= 1:
                    while len(parents_pool) < 3: # Garante ao menos 3 pais (podem ser repetidos)
                        parents_pool.append(random.choice(available_parents))
                else: # Fallback extremo: nenhuma avaliação bem-sucedida
                    child = create_individual()
                    if progress_callback:
                        progress_callback({"type": "warning", "message": "  Aviso: Nenhuma avaliação bem-sucedida. Criando indivíduo aleatório para nova população."})
                    new_population.append(child)
                    continue # Pula para a próxima iteração do while
            elif len(parents_pool) == 0: # Caso evaluations esteja vazio (ex: tudo falhou)
                child = create_individual()
                if progress_callback:
                    progress_callback({"type": "warning", "message": "  Aviso: Nenhuma avaliação bem-sucedida. Criando indivíduo aleatório para nova população."})
                new_population.append(child)
                continue
            
            p1, p2, p3 = random.sample(parents_pool, 3) # Agora deve ter pelo menos 3 pais no pool

            child = crossover(p1, p2, p3)

            if random.random() < mutation_rate:
                filho = mutate(child)
            new_population.append(child)

        population = new_population
        if progress_callback:
            progress_callback({"type": "info", "message": f"  População para a próxima geração criada."})


    total_time = time.time() - start_time
    
    # Obter exemplos de imagem apenas no final
    # Apenas se não foi interrompido e temos um melhor modelo
    if not interrupted and best_preds_final is not None:
        correct_examples = get_image_examples_for_frontend(full_val_set, best_preds_final, best_labels_final, acertos=True, n=5)
        incorrect_examples = get_image_examples_for_frontend(full_val_set, best_preds_final, best_labels_final, acertos=False, n=5)
    else:
        correct_examples = []
        incorrect_examples = []

    final_results = {
        "melhor_individuo": best_individual,
        "melhor_acuracia": best_accuracy,
        "melhor_precisao": best_precision,
        "historico_acuracias": history_accuracies,
        "tempo_total_segundos": total_time,
        "correct_examples": correct_examples,
        "incorrect_examples": incorrect_examples,
        "interrupted": interrupted # Adiciona flag de interrupção aos resultados finais
    }

    if progress_callback:
        progress_callback({"type": "final_results", "data": final_results})
        if interrupted:
            progress_callback({"type": "info", "message": "Algoritmo Genético interrompido. Resultados parciais enviados."})
        else:
            progress_callback({"type": "info", "message": "Algoritmo Genético concluído. Resultados finais enviados."})

    return final_results

if __name__ == '__main__':
    # Exemplo de como rodar o AG diretamente (para testes)
    def print_progress(message):
        print(message)

    # Crie um evento de parada para o exemplo local
    stop_event_example = asyncio.Event()

    params = {
        "pop_size": 6,
        "generations": 2,
        "mutation_rate": 0.3,
        "ag_epochs": 2,
        "learning_rate_options": [0.01, 0.005],
        "batch_size_options": [8, 16],
        "n_filters_options": [16, 32],
        "n_fc_options": [64, 128],
        "dropout_options": [0.0, 0.25]
    }
    
    # Para testar a interrupção localmente:
    # asyncio.get_event_loop().call_later(5, stop_event_example.set) # Interrompe após 5 segundos
    
    results = run_genetic_algorithm(**params, progress_callback=print_progress, stop_event_flag=stop_event_example)
    print("\nResultados Finais:", results)