import os
import pm4py
import rustxes
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.conversion.bpmn import converter as bpmn_converter
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
from pm4py.algo.discovery.footprints import algorithm as footprints_discovery
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.evaluation.precision import algorithm as precision_algo
from pm4py.algo.evaluation.generalization import algorithm as generalization_algo
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_algo
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import permutations
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import psutil
import logging
import gc

# Otimizações adicionadas
import pickle
import hashlib
import numpy as np
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    jit = lambda *args, **kwargs: lambda f: f

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========================= PASTA OUTPUT =========================
OUTPUT_DIR = "OUTPUT"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    logger.info(f"Pasta OUTPUT criada em: {OUTPUT_DIR}")

# ========================= CACHE SYSTEM =========================
class BPMNModelCache:
    """Sistema de cache para modelos BPMN convertidos em Petri Nets."""
    
    def __init__(self, cache_dir=".cache", max_cache_items=20):
        self.cache_dir = cache_dir
        self.max_cache_items = max_cache_items
        self.cache_count = 0
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def _get_cache_key(self, filepath):
        """Gera uma chave de cache baseada no caminho e conteúdo do arquivo."""
        try:
            with open(filepath, 'rb') as f:
                content_hash = hashlib.md5(f.read()).hexdigest()
            return content_hash
        except:
            return hashlib.md5(filepath.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key):
        """Retorna o caminho do arquivo de cache."""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def get(self, filepath):
        """Carrega modelo do cache se existir, senão retorna None."""
        try:
            cache_key = self._get_cache_key(filepath)
            cache_path = self._get_cache_path(cache_key)
            
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    logger.debug(f"✓ Cache hit para {os.path.basename(filepath)}")
                    return pickle.load(f)
        except Exception as e:
            logger.debug(f"Cache miss para {os.path.basename(filepath)}: {e}")
        
        return None
    
    def set(self, filepath, model_tuple):
        """Salva modelo no cache."""
        try:
            # Limitar tamanho do cache
            if self.cache_count >= self.max_cache_items:
                self.clear_cache()
            
            cache_key = self._get_cache_key(filepath)
            cache_path = self._get_cache_path(cache_key)
            
            with open(cache_path, 'wb') as f:
                pickle.dump(model_tuple, f)
                logger.debug(f"✓ Cache salvo para {os.path.basename(filepath)}")
                self.cache_count += 1
        except Exception as e:
            logger.warning(f"Erro ao salvar cache: {e}")
    
    def clear_cache(self):
        """Limpa todo o cache para liberar memória."""
        try:
            if os.path.exists(self.cache_dir):
                for file in os.listdir(self.cache_dir):
                    file_path = os.path.join(self.cache_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                self.cache_count = 0
                logger.info("Cache limpo para liberar memória")
        except Exception as e:
            logger.warning(f"Erro ao limpar cache: {e}")

# Instância global de cache
model_cache = BPMNModelCache()

# ========================= OTIMIZAÇÕES NUMBA =========================
@jit(nopython=False) if HAS_NUMBA else (lambda f: f)
def compute_fitness_score(trace_fitnesses):
    """Calcula score de fitness de forma otimizada."""
    if len(trace_fitnesses) == 0:
        return 0.0
    return float(np.sum(trace_fitnesses)) / len(trace_fitnesses)

def get_trace_fitnesses_optimized(token_replay_results):
    """Extrai fitness das traces de forma otimizada."""
    try:
        fitnesses = [res.get("trace_fitness", 0.0) for res in token_replay_results]
        return np.array(fitnesses, dtype=np.float32)
    except:
        return np.array([res["trace_fitness"] for res in token_replay_results], dtype=np.float32)

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_system_info():
    """Exibe informacoes do sistema."""
    print("\n" + "="*60)
    print("INFORMACOES DO SISTEMA")
    print("="*60)
    print(f"CPUs disponiveis: {psutil.cpu_count()}")
    print(f"CPUs logicas: {psutil.cpu_count(logical=True)}")
    print(f"CPUs fisicas: {psutil.cpu_count(logical=False)}")
    print(f"CPU atual (%): {psutil.cpu_percent(interval=1)}")
    
    # Mostrar uso por núcleo
    cpu_per_core = psutil.cpu_percent(interval=0.5, percpu=True)
    print(f"\nUSO POR NÚCLEO (primeiros 10):")
    for i, percent in enumerate(cpu_per_core[:10]):
        print(f"  Núcleo {i}: {percent}%")
    if len(cpu_per_core) > 10:
        print(f"  ... e mais {len(cpu_per_core) - 10} núcleos")
    
    print(f"\nMemoria disponivel: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print(f"Memoria usada (%): {psutil.virtual_memory().percent}%")
    
    print("\nOTIMIZACOES ATIVAS:")
    print(f"  - Cache de modelos BPMN: ATIVADO")
    print(f"  - Compilacao Numba JIT: {'ATIVADO' if HAS_NUMBA else 'NAO DISPONIVEL'}")
    print(f"  - Processamento com numpy: ATIVADO")
    print(f"  - Multithreading/Multiprocessing paralelo: ATIVADO")
    print("="*60 + "\n")

def evaluate_model(log, petri_net, initial_marking, final_marking):
    """
    Avalia um modelo usando pm4py otimizado.
    """
    try:
        token_replay_results = token_replay.apply(log, petri_net, initial_marking, final_marking)
        
        # Usar versão otimizada com numpy
        trace_fitnesses = get_trace_fitnesses_optimized(token_replay_results)
        recall = compute_fitness_score(trace_fitnesses)

        precision = precision_algo.apply(log, petri_net, initial_marking, final_marking)
        generalization = generalization_algo.apply(log, petri_net, initial_marking, final_marking)
        simplicity = simplicity_algo.apply(petri_net)
        
        return recall, precision, generalization, simplicity
    finally:
        # Liberar memória explicitamente
        del token_replay_results, trace_fitnesses
        gc.collect()

def evaluate_model_with_memory_optimization(log, petri_net, initial_marking, final_marking):
    """
    Versão otimizada de memória (reduz cópia de dados e usa numpy).
    """
    token_replay_results = None
    trace_fitnesses = None
    try:
        # Usar variants otimizados quando disponíveis
        token_replay_results = token_replay.apply(
            log, petri_net, initial_marking, final_marking,
            variant=token_replay.Variants.TOKEN_REPLAY
        )
        
        # Usar versão otimizada com numpy
        trace_fitnesses = get_trace_fitnesses_optimized(token_replay_results)
        recall = compute_fitness_score(trace_fitnesses)

        precision = precision_algo.apply(log, petri_net, initial_marking, final_marking)
        generalization = generalization_algo.apply(log, petri_net, initial_marking, final_marking)
        simplicity = simplicity_algo.apply(petri_net)

        return recall, precision, generalization, simplicity
    except Exception as e:
        logger.warning(f"Erro na avaliação otimizada, usando modo padrão: {e}")
        return evaluate_model(log, petri_net, initial_marking, final_marking)
    finally:
        # Liberar memória explicitamente
        if token_replay_results is not None:
            del token_replay_results
        if trace_fitnesses is not None:
            del trace_fitnesses
        gc.collect()

def evaluate_solution_model(solution_file, reference_folder_path, reference_log, use_memory_optimization=True, use_cache=False):
    """
    Avalia um modelo de solução individual.
    Retorna uma tupla (solution_file, result, execution_time) ou (solution_file, None, None) em caso de erro.
    """
    bpmn_graph_s = None
    sol_net = None
    sol_im = None
    sol_fm = None
    try:
        solution_path = os.path.join(reference_folder_path, solution_file)
        
        # Só usar cache se explicitamente solicitado
        if use_cache:
            cached_model = model_cache.get(solution_path)
            if cached_model is not None:
                sol_net, sol_im, sol_fm = cached_model
        
        if sol_net is None:
            # Carregar e converter BPMN
            bpmn_graph_s = pm4py.read_bpmn(solution_path)
            sol_net, sol_im, sol_fm = bpmn_converter.apply(bpmn_graph_s, variant=bpmn_converter.Variants.TO_PETRI_NET)
            
            # Liberar BPMN graph imediatamente
            del bpmn_graph_s
            bpmn_graph_s = None
            gc.collect()

        start_time = time.time()
        
        # Usar versão otimizada se solicitado
        if use_memory_optimization:
            recall, precision, generalization, simplicity = evaluate_model_with_memory_optimization(
                reference_log, sol_net, sol_im, sol_fm
            )
        else:
            recall, precision, generalization, simplicity = evaluate_model(
                reference_log, sol_net, sol_im, sol_fm
            )
        
        end_time = time.time()

        execution_time = end_time - start_time

        # Copiar apenas os valores (não manter referências)
        result = {
            "model": str(solution_file),
            "recall": float(recall),
            "precision": float(precision),
            "generalization": float(generalization),
            "simplicity": float(simplicity),
        }
        
        # Liberar TODA a memória imediatamente
        del recall, precision, generalization, simplicity
        if sol_net is not None:
            del sol_net
        if sol_im is not None:
            del sol_im
        if sol_fm is not None:
            del sol_fm
        gc.collect()
        
        return solution_file, result, execution_time
    except Exception as e:
        logger.error(f"Erro ao processar {solution_file}: {str(e)}")
        return solution_file, None, None
    finally:
        # Liberar memória explicitamente (não manter referências)
        if bpmn_graph_s is not None:
            del bpmn_graph_s
        if 'sol_net' in locals() and sol_net is not None:
            del sol_net
        if 'sol_im' in locals() and sol_im is not None:
            del sol_im
        if 'sol_fm' in locals() and sol_fm is not None:
            del sol_fm
        gc.collect()

def process_folder_with_multithreading(reference_folder_path, reference_log, max_workers=4, use_memory_optimization=True, max_arch=5):
    """
    Processa modelos em LOTES com processamento paralelo DENTRO de cada lote.
    Após cada lote, LIBERA TODA a memória antes de processar o próximo.
    
    Args:
        reference_folder_path: Caminho para a pasta contendo os modelos
        reference_log: Log de referência para avaliação
        max_workers: Número máximo de threads/núcleos para usar EM PARALELO (max_process)
        use_memory_optimization: Se True, usa versão otimizada de memória
        max_arch: Tamanho do lote (quantos arquivos processar por vez)
    
    Returns:
        Tupla (folder_results, execution_times)
    """
    folder_results = []
    execution_times = []
    
    # Encontrar todos os arquivos de solução
    solution_files = [f for f in os.listdir(reference_folder_path) 
                      if f.startswith("S") and f.endswith(".bpmn")]
    
    if not solution_files:
        return folder_results, execution_times
    
    # Ordenar por tamanho e INTERCALAR (balanceamento de carga)
    # Em vez de: pequeno, pequeno, pequeno, grande, grande, grande
    # Fazer: pequeno, grande, médio, grande, pequeno, médio...
    solution_files_with_size = [
        (f, os.path.getsize(os.path.join(reference_folder_path, f)))
        for f in solution_files
    ]
    solution_files_with_size.sort(key=lambda x: x[1])
    
    # Intercalar: distribuir arquivos pequenos e grandes nos lotes
    # Isso evita que um lote tenha só arquivos grandes (lento) e outro só pequenos (rápido)
    total = len(solution_files_with_size)
    balanced_files = []
    left, right = 0, total - 1
    use_left = True
    
    while left <= right:
        if use_left:
            balanced_files.append(solution_files_with_size[left][0])
            left += 1
        else:
            balanced_files.append(solution_files_with_size[right][0])
            right -= 1
        use_left = not use_left
    
    solution_files = balanced_files
    
    total_models = len(solution_files)
    logger.info(f"  Encontrados {total_models} modelos")
    logger.info(f"  Estratégia: Lotes de {max_arch} arquivos com {max_workers} threads em paralelo")
    logger.info(f"  Total de lotes: {(total_models + max_arch - 1) // max_arch}")
    
    # Processar em LOTES
    completed = 0
    
    for batch_num, i in enumerate(range(0, total_models, max_arch), start=1):
        batch = solution_files[i:i+max_arch]
        batch_size = len(batch)
        
        # Calcular quantos núcleos serão realmente usados
        cores_used = min(batch_size, max_workers)
        cores_idle = max(0, max_workers - batch_size)
        utilization = (cores_used / max_workers) * 100 if max_workers > 0 else 0
        
        logger.info(f"\n  ► LOTE {batch_num}/{(total_models + max_arch - 1) // max_arch} - {batch_size} arquivos")
        logger.info(f"  → Núcleos: {cores_used}/{max_workers} ativos ({utilization:.1f}% utilização)" + 
                   (f" - {cores_idle} ociosos" if cores_idle > 0 else " - 100% otimizado!"))
        batch_start_time = time.time()
        
        # Processar o LOTE atual em PARALELO com max_workers threads
        batch_results = []
        batch_times = []
        
        # Diagnóstico: contar threads ativas
        import threading
        threads_before = threading.active_count()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submeter todos os arquivos do LOTE para processamento paralelo
            futures = {
                executor.submit(evaluate_solution_model, sol_file, reference_folder_path, 
                              reference_log, use_memory_optimization, False): sol_file
                for sol_file in batch
            }
            
            # Verificar quantas threads foram criadas
            threads_after = threading.active_count()
            threads_created = threads_after - threads_before
            logger.info(f"  → Threads criadas: {threads_created} (esperado: {min(batch_size, max_workers)})")
            
            if threads_created < max_workers and batch_size >= max_workers:
                logger.warning(f"  ⚠ Apenas {threads_created} threads ativas! Python pode estar limitando por GIL.")
                logger.warning(f"  → Considere usar multiprocessing em vez de threads.")
            
            # Coletar resultados conforme cada thread do lote termina
            for future in as_completed(futures):
                try:
                    solution_file, result, exec_time = future.result()
                    
                    if result is not None:
                        batch_results.append(result)
                        batch_times.append(exec_time)
                        completed += 1
                        
                        # Monitorar CPU a cada 5 arquivos
                        if completed % 5 == 0:
                            cpu_percent = psutil.cpu_percent(interval=0.1, percpu=False)
                            logger.info(f"    [{completed}/{total_models}] {solution_file} ({exec_time:.2f}s) - CPU: {cpu_percent}%")
                        else:
                            logger.info(f"    [{completed}/{total_models}] {solution_file} ({exec_time:.2f}s)")
                    
                    # Liberar memória deste item
                    del result, exec_time, solution_file
                    
                finally:
                    del future
                    gc.collect()
        
        # Adicionar resultados do lote aos resultados gerais
        folder_results.extend(batch_results)
        execution_times.extend(batch_times)
        
        # LIBERAÇÃO TOTAL DE MEMÓRIA após o lote
        batch_end_time = time.time()
        logger.info(f"  ✓ Lote {batch_num} concluído em {batch_end_time - batch_start_time:.2f}s")
        
        # Deletar TUDO do lote
        del futures, batch, batch_results, batch_times
        
        # Forçar garbage collection AGRESSIVO
        gc.collect()
        gc.collect()  # Duas vezes para garantir
        
        # Mostrar memória disponível
        mem_info = psutil.virtual_memory()
        logger.info(f"  → Memória liberada: {mem_info.available / (1024**3):.2f} GB disponíveis ({100-mem_info.percent:.1f}% livre)\n")
    
    return folder_results, execution_times

def process_folder_with_multiprocessing(reference_folder_path, reference_log, max_workers=4, use_memory_optimization=True, max_arch=5):
    """
    Processa modelos em LOTES com processamento paralelo DENTRO de cada lote (multiprocessing).
    Após cada lote, LIBERA TODA a memória antes de processar o próximo.
    """
    folder_results = []
    execution_times = []
    
    # Encontrar todos os arquivos de solução
    solution_files = [f for f in os.listdir(reference_folder_path) 
                      if f.startswith("S") and f.endswith(".bpmn")]
    
    if not solution_files:
        return folder_results, execution_times
    
    # Ordenar por tamanho e INTERCALAR (balanceamento de carga)
    solution_files_with_size = [
        (f, os.path.getsize(os.path.join(reference_folder_path, f)))
        for f in solution_files
    ]
    solution_files_with_size.sort(key=lambda x: x[1])
    
    # Intercalar: distribuir arquivos pequenos e grandes nos lotes
    total = len(solution_files_with_size)
    balanced_files = []
    left, right = 0, total - 1
    use_left = True
    
    while left <= right:
        if use_left:
            balanced_files.append(solution_files_with_size[left][0])
            left += 1
        else:
            balanced_files.append(solution_files_with_size[right][0])
            right -= 1
        use_left = not use_left
    
    solution_files = balanced_files
    
    total_models = len(solution_files)
    logger.warning("Usando multiprocessing (mais lento em Windows)")
    logger.info(f"  Encontrados {total_models} modelos")
    logger.info(f"  Estratégia: Lotes de {max_arch} arquivos com {max_workers} processos em paralelo")
    logger.info(f"  Total de lotes: {(total_models + max_arch - 1) // max_arch}")
    
    # Processar em LOTES
    completed = 0
    
    for batch_num, i in enumerate(range(0, total_models, max_arch), start=1):
        batch = solution_files[i:i+max_arch]
        batch_size = len(batch)
        
        # Calcular quantos núcleos serão realmente usados
        cores_used = min(batch_size, max_workers)
        cores_idle = max(0, max_workers - batch_size)
        utilization = (cores_used / max_workers) * 100 if max_workers > 0 else 0
        
        logger.info(f"\n  ► LOTE {batch_num}/{(total_models + max_arch - 1) // max_arch} - {batch_size} arquivos")
        logger.info(f"  → Núcleos: {cores_used}/{max_workers} ativos ({utilization:.1f}% utilização)" + 
                   (f" - {cores_idle} ociosos" if cores_idle > 0 else " - 100% otimizado!"))
        batch_start_time = time.time()
        
        batch_results = []
        batch_times = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submeter todos do lote
            futures = {
                executor.submit(evaluate_solution_model, sol_file, reference_folder_path, 
                              reference_log, use_memory_optimization, False): sol_file
                for sol_file in batch
            }
            
            logger.info(f"  → Processos criados: até {max_workers} (depende do SO)")
            
            for future in as_completed(futures):
                try:
                    solution_file, result, exec_time = future.result()
                    
                    if result is not None:
                        batch_results.append(result)
                        batch_times.append(exec_time)
                        completed += 1
                        logger.info(f"    [{completed}/{total_models}] {solution_file} ({exec_time:.2f}s)")
                    
                    del result, exec_time, solution_file
                    
                finally:
                    del future
                    gc.collect()
        
        folder_results.extend(batch_results)
        execution_times.extend(batch_times)
        
        batch_end_time = time.time()
        logger.info(f"  ✓ Lote {batch_num} concluído em {batch_end_time - batch_start_time:.2f}s")
        
        del futures, batch, batch_results, batch_times
        gc.collect()
        gc.collect()
        
        mem_info = psutil.virtual_memory()
        logger.info(f"  → Memória liberada: {mem_info.available / (1024**3):.2f} GB disponíveis ({100-mem_info.percent:.1f}% livre)\n")
    
    return folder_results, execution_times

def plot_results_per_metric(results, folder_name):
    """Plota resultados por metrica e salva em OUTPUT."""
    metrics = ['Recall', 'Precision', 'Generalization', 'Simplicity']
    models = [result["model"] for result in results]
    values = {
        'Recall': [result["recall"] for result in results],
        'Precision': [result["precision"] for result in results],
        'Generalization': [result["generalization"] for result in results],
        'Simplicity': [result["simplicity"] for result in results]
    }

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        print(f"Total modelos: {range(1, len(models)+1)}")
        ax.plot(range(1, len(models)+1), values[metric], marker='o', linestyle='-', label=metric, color=colors[i])
        ax.set_title(f'{metric}', fontsize=10)
        ax.set_xlabel(f'Model', labelpad=0.05, fontsize=8)
        ax.set_ylabel('Value', labelpad=0.05, fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', rotation=45)
        ax.legend()

        # Decrease font size of axis value labels
        ax.tick_params(axis='x', labelsize=5)  # Decrease x-axis value font size
        ax.tick_params(axis='y', labelsize=5)  # Decrease y-axis value font size
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.suptitle(f'Results for Folder: {folder_name}', y=1.02)
    
    # Salvar grafico
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUTPUT_DIR, f"01_ResultsPerMetric_{folder_name}_{timestamp}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"Grafico salvo: {filename}")
    plt.close(fig)
    plt.clf()
    
    # Liberar memória de variáveis grandes
    del models, values, fig, axes
    gc.collect()

def plot_box_plots(global_results, folder_solution_counts):
    """Plota box plots comparativos e salva em OUTPUT."""
    metrics = ['Recall', 'Precision', 'Generalization', 'Simplicity']
    global_data = {metric: [] for metric in metrics}
    folders = []

    # Sort folders by solution count
    sorted_folders = sorted(folder_solution_counts.items(), key=lambda x: x[1])

    for folder_name, _ in sorted_folders:
        for result in global_results[folder_name]:
            for metric in metrics:
                global_data[metric].append(result[metric.lower()])
        folders.extend([str(folder_solution_counts[folder_name])] * len(global_results[folder_name]))

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 6), sharey=True)

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        sns.boxplot(x=folders, y=global_data[metric], ax=ax, palette="Set3")
        ax.set_title(metric, fontsize=14)
        ax.set_xlabel('Number of Solutions', fontsize=10)
        ax.set_ylabel('Value', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.suptitle('Global Metrics Comparison', y=1.02)
    
    # Salvar grafico
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUTPUT_DIR, f"02_BoxPlots_GlobalComparison_{timestamp}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"Grafico salvo: {filename}")
    plt.close(fig)
    plt.clf()
    
    # Liberar memória de variáveis grandes
    del global_data, folders, fig, axes
    gc.collect()

def plot_execution_times(global_execution_times, folder_solution_counts):
    """Plota tempos de execucao e salva em OUTPUT."""
    execution_data = []
    labels = []

    # Sort folders by solution count
    sorted_folders = sorted(folder_solution_counts.items(), key=lambda x: x[1])

    for folder_name, _ in sorted_folders:
        execution_data.extend(global_execution_times[folder_name])
        solution_count = folder_solution_counts.get(folder_name, 0)
        labels.extend([str(solution_count)] * len(global_execution_times[folder_name]))

    plt.figure(figsize=(12, 6))
    sns.boxplot(x=labels, y=execution_data, palette="Set3")
    plt.title("Execution Time vs Number of Solutions", fontsize=16)
    plt.xlabel("Number of Solutions", fontsize=12)
    plt.ylabel("Execution Time (s)", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    
    # Salvar grafico
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUTPUT_DIR, f"03_ExecutionTimes_{timestamp}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"Grafico salvo: {filename}")
    plt.close()
    plt.clf()
    
    # Liberar memória de variáveis grandes
    del execution_data, labels
    gc.collect()

def display_solution_counts_table(folder_solution_counts):
    """Displays a table showing the count of solutions per folder and saves to CSV."""
    folder_indices = list(range(1, len(folder_solution_counts) + 1))
    data = {
        "Folder Index": folder_indices,
        "Folder Name": list(folder_solution_counts.keys()),
        "Solution Count": list(folder_solution_counts.values())
    }
    df = pd.DataFrame(data)
    print("\n=== Solution Counts by Folder ===")
    print(df.to_string(index=False))
    
    # Salvar CSV
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUTPUT_DIR, f"01_SolutionCounts_{timestamp}.csv")
    df.to_csv(filename, index=False)
    logger.info(f"Tabela salva: {filename}")
    
    return df

def display_experiment_results_table(global_results):
    """Displays a table summarizing the results of each experiment and saves to CSV."""
    rows = []
    for folder_name, results in global_results.items():
        for result in results:
            rows.append({
                "Folder Name": folder_name,
                "Model": result["model"],
                "Recall": result["recall"],
                "Precision": result["precision"],
                "Generalization": result["generalization"],
                "Simplicity": result["simplicity"]
            })
    df = pd.DataFrame(rows)
    print("\n=== Experiment Results ===")
    print(df.to_string(index=False))
    
    # Salvar CSV
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUTPUT_DIR, f"02_ExperimentResults_{timestamp}.csv")
    df.to_csv(filename, index=False)
    logger.info(f"Tabela salva: {filename}")
    
    return df

def display_metric_statistics(global_results, folder_solution_counts):
    """Displays a table with mean and standard deviation of each metric for each folder and saves to CSV."""
    rows = []
    for folder_index, (folder_name, results) in enumerate(global_results.items(), start=1):
        recalls = [res["recall"] for res in results]
        precisions = [res["precision"] for res in results]
        generalizations = [res["generalization"] for res in results]
        simplicities = [res["simplicity"] for res in results]

        # Compute mean and standard deviation for each metric
        row = {
            "Folder Index": folder_index,
            "Folder Name": folder_name,
            "Solution Count": folder_solution_counts[folder_name],
            "Recall Mean": round(pd.Series(recalls).mean(), 4),
            "Recall Std Dev": round(pd.Series(recalls).std(), 4),
            "Precision Mean": round(pd.Series(precisions).mean(), 4),
            "Precision Std Dev": round(pd.Series(precisions).std(), 4),
            "Generalization Mean": round(pd.Series(generalizations).mean(), 4),
            "Generalization Std Dev": round(pd.Series(generalizations).std(), 4),
            "Simplicity Mean": round(pd.Series(simplicities).mean(), 4),
            "Simplicity Std Dev": round(pd.Series(simplicities).std(), 4),
        }
        rows.append(row)

    # Convert rows into a DataFrame
    df = pd.DataFrame(rows)
    print("\n=== Metric Statistics (Mean and Std Dev) ===")
    print(df.to_string(index=False))
    
    # Salvar CSV
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUTPUT_DIR, f"03_MetricStatistics_{timestamp}.csv")
    df.to_csv(filename, index=False)
    logger.info(f"Tabela salva: {filename}")
    
    return df

def main():
    print("\n" + "="*80)
    print("AVALIACAO DE QUALIDADE DE MODELOS BPMN")
    print("="*80 + "\n")
    
    # Tempo de inicio geral
    script_start_time = time.time()
    script_start_timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Tempo de inicio do script: {script_start_timestamp}\n")
    
    print_system_info()
    
    main_folder = input(" Caminho da pasta principal com modelos: ")
    
    # max_process: Quantos núcleos/threads usar em paralelo
    cpu_count = psutil.cpu_count()
    max_process_input = input(f"  Número de núcleos/threads (max_process, padrao {cpu_count}): ").strip()
    max_process = int(max_process_input) if max_process_input.isdigit() and int(max_process_input) > 0 else cpu_count
    
    # max_arch: Quantos arquivos processar por lote
    print(f"\n  IMPORTANTE: max_arch controla MEMÓRIA (arquivos por lote)")
    print(f"              max_process controla CPU (núcleos em paralelo)")
    print(f"              Se max_arch < max_process, alguns núcleos ficarão ociosos")
    print(f"              Se max_arch muito alto, pode consumir muita memória!\n")
    
    suggested_max_arch = 10  # Sugestão conservadora
    max_arch_input = input(f"  Tamanho do lote (max_arch - arquivos por vez, padrao {suggested_max_arch}): ").strip()
    max_arch = int(max_arch_input) if max_arch_input.isdigit() and int(max_arch_input) > 0 else suggested_max_arch
    
    # Apenas AVISAR, não forçar ajuste
    if max_arch < max_process:
        logger.warning(f"\n  ⚠ AVISO: max_arch ({max_arch}) < max_process ({max_process})")
        logger.warning(f"  → Apenas {max_arch} núcleos trabalharão por vez ({max_process - max_arch} ociosos)")
        logger.warning(f"  → Para usar todos os núcleos, aumente max_arch para {max_process}")
        logger.warning(f"  → Mas CUIDADO: mais arquivos = mais memória!\n")
        
        adjust = input(f"  Deseja ajustar max_arch para {max_process}? (s/n): ").strip().lower()
        if adjust == 's':
            max_arch = max_process
            logger.info(f"  ✓ max_arch ajustado para {max_arch}\n")
        else:
            logger.info(f"  ✓ Mantendo max_arch = {max_arch} (prioriza memória sobre CPU)\n")
    
    use_optimization = input(" Usar otimizacao de memoria? (s/n, padrao s): ").strip().lower() != 'n'
    
    executor_type = input(" Usar threads (t) ou multiprocessing (m)? (padrao m): ").strip().lower()
    if executor_type == 't':
        logger.warning("\n  ⚠ THREADS podem não usar todos os núcleos devido ao Python GIL!")
        logger.warning("  Recomendamos MULTIPROCESSING para melhor distribuição de carga.\n")
        use_multiprocessing = False
    else:
        logger.info("\n  ✓ MULTIPROCESSING: Melhor distribuição de CPU entre núcleos\n")
        use_multiprocessing = True

    global_results = {}
    folder_solution_counts = {}
    global_execution_times = {}
    total_start = time.time()

    for reference_model_folder in os.listdir(main_folder):
        reference_folder_path = os.path.join(main_folder, reference_model_folder)
        if os.path.isdir(reference_folder_path):
            rm_file = None
            xes_file = None
            for file in os.listdir(reference_folder_path):
                if file.endswith(".xes"):
                    xes_file = file
                    break
                else:
                    if file.startswith("RM") and file.endswith(".bpmn"):
                        rm_file = file
                        break
            reference_log = None
            bpmn_graph = None
            ref_net = None
            ref_im = None
            ref_fm = None

            if xes_file:
                reference_path = os.path.join(reference_folder_path, xes_file)
                reference_log = pm4py.read_xes(reference_path)
            else:
                if rm_file:
                    reference_path = os.path.join(reference_folder_path, rm_file)
                    bpmn_graph = pm4py.read_bpmn(reference_path)
                    ref_net, ref_im, ref_fm = bpmn_converter.apply(bpmn_graph, variant=bpmn_converter.Variants.TO_PETRI_NET)
                    reference_log = simulator.apply(ref_net, ref_im, variant=simulator.Variants.BASIC_PLAYOUT, parameters={"no_traces": 1})
                    # Liberar BPMN e Petri net após gerar o log
                    del bpmn_graph, ref_net, ref_im, ref_fm
                    gc.collect()

            if reference_log is not None and len(reference_log) > 0:
                executor_name = "Multiprocessing" if use_multiprocessing else "Multithreading"
                logger.info(f"\n Processando: {reference_model_folder}")
                logger.info(f"   Executor: {executor_name}")
                logger.info(f"   Lotes: {max_arch} arquivos por lote")
                logger.info(f"   Paralelismo: {max_process} threads/núcleos")
                logger.info(f"   Otimização: {'SIM' if use_optimization else 'NAO'}")
                
                # Usar multithreading ou multiprocessing para processar modelos
                if use_multiprocessing:
                    folder_results, execution_times = process_folder_with_multiprocessing(
                        reference_folder_path, reference_log, max_workers=max_process,
                        use_memory_optimization=use_optimization, max_arch=max_arch
                    )
                else:
                    folder_results, execution_times = process_folder_with_multithreading(
                        reference_folder_path, reference_log, max_workers=max_process,
                        use_memory_optimization=use_optimization, max_arch=max_arch
                    )

                if folder_results:
                    logger.info(f"OK {len(folder_results)} modelos processados")
                    plot_results_per_metric(folder_results, reference_model_folder)

                global_results[reference_model_folder] = folder_results
                folder_solution_counts[reference_model_folder] = len(folder_results)
                global_execution_times[reference_model_folder] = execution_times
                
                # Limpar variáveis após adicionar aos resultados globais
                del folder_results
                del execution_times
            
            # LIMPAR CACHE após cada pasta para liberar memória
            model_cache.clear_cache()
            
            # Liberar reference_log e outras variáveis após processar a pasta
            if reference_log is not None:
                del reference_log
            if 'bpmn_graph' in locals() and bpmn_graph is not None:
                del bpmn_graph
            if 'ref_net' in locals() and ref_net is not None:
                del ref_net
            if 'ref_im' in locals() and ref_im is not None:
                del ref_im
            if 'ref_fm' in locals() and ref_fm is not None:
                del ref_fm
            
            # Forçar garbage collection AGRESSIVO após cada pasta
            gc.collect()
            gc.collect()  # Chamar duas vezes para garantir
            
            # Log de memória
            mem_info = psutil.virtual_memory()
            logger.info(f"  Memória disponível: {mem_info.available / (1024**3):.2f} GB ({100-mem_info.percent:.1f}% livre)")

    total_end = time.time()
    logger.info(f"\n  Tempo total de execucao: {total_end - total_start:.2f}s")

    if global_results:
        plot_box_plots(global_results, folder_solution_counts)
        
    if global_execution_times:
        plot_execution_times(global_execution_times, folder_solution_counts)

    # Display the solution counts table
    solution_counts_df = display_solution_counts_table(folder_solution_counts)

    # Display the experiment results table
    experiment_results_df = display_experiment_results_table(global_results)

    # Display the metric statistics table
    metric_statistics_df = display_metric_statistics(global_results, folder_solution_counts)
    
    # Resumo final
    print("\n" + "="*70)
    print("RESUMO FINAL")
    print("="*70)
    
    # Tempo final geral
    script_end_time = time.time()
    script_end_timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    script_duration = script_end_time - script_start_time
    
    print(f"Tempo final do script: {script_end_timestamp}")
    print(f"Duracao total: {script_duration/60:.2f} minutos ({script_duration:.2f}s)")
    print(f"Tempo total de analise: {(total_end - total_start)/60:.2f} minutos")
    print(f"Graficos salvos em: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Arquivos gerados:")
    for file in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  - {file}")
    print("="*70 + "\n")

    return solution_counts_df, experiment_results_df, metric_statistics_df

if __name__ == "__main__":
    # main()
    solution_counts_table, experiment_results_table, metric_statistics_table = main()
