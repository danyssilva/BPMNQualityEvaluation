"""
Script otimizado para avaliar APENAS os algoritmos clássicos usando CSVs já gerados.

Este script:
1. Lê os CSVs já existentes da avaliação de seus modelos BPMN
2. Procura automaticamente pelos arquivos XES em cada pasta de processo
3. Executa APENAS a comparação com algoritmos clássicos
4. Gera visualizações comparativas

USO:
    python evaluate_only_classic_algorithms.py --csv OUTPUT/02_ExperimentResults_20260115_212552.csv --input INPUT/EXPERIMENTOS_SEMRM --output OUTPUT/COMPARISON_RESULTS

Benefício: Muito mais rápido, sem duplicar o trabalho de avaliação de modelos.
Cada processo pode ter seu próprio arquivo XES que será detectado automaticamente.
"""

import os
import sys
import pandas as pd
import argparse
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import traceback
import numpy as np
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    jit = lambda *args, **kwargs: lambda f: f

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from compare_mining_algorithms import batch_compare_all_processes, compare_process_algorithms


def find_xes_files(input_folder):
    """
    Procura recursivamente por arquivos .xes em cada pasta de processo.
    Retorna um dicionário: {processo: caminho_do_xes}
    """
    xes_files = {}
    
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.xes'):
                # Usa o nome da pasta do processo como chave
                process_name = os.path.basename(root)
                xes_path = os.path.join(root, file)
                
                if process_name not in xes_files:
                    xes_files[process_name] = xes_path
                else:
                    logger.warning(f"Multiplos XES encontrados em {process_name}, usando: {xes_path}")
                    xes_files[process_name] = xes_path
    
    logger.info(f"\nEncontrados {len(xes_files)} arquivos XES:")
    for process, path in sorted(xes_files.items()):
        logger.info(f"  - {process}: {os.path.basename(path)}")
    
    return xes_files


def get_latest_metric_csv(csv_dir="OUTPUT/OUTPUT_EXPERIMENTOS_COMRM"):
    """
    Encontra o CSV de estatísticas de métricas mais recente (03_MetricStatistics_*.csv)
    """
    if not os.path.exists(csv_dir):
        raise FileNotFoundError(f"Pasta nao encontrada: {csv_dir}")
    
    csv_files = [f for f in os.listdir(csv_dir) if f.startswith("03_MetricStatistics") and f.endswith(".csv")]
    
    if not csv_files:
        raise FileNotFoundError(f"Nenhum arquivo 03_MetricStatistics_*.csv encontrado em {csv_dir}")
    
    csv_files.sort(reverse=True)
    latest = os.path.join(csv_dir, csv_files[0])
    logger.info(f"Usando CSV de metricas: {latest}")
    return latest


def get_latest_csv(csv_dir="OUTPUT/OUTPUT_EXPERIMENTOS_COMRM"):
    """
    Encontra o CSV mais recente na pasta especificada se não foi especificado.
    """
    if not os.path.exists(csv_dir):
        raise FileNotFoundError(f"Pasta nao encontrada: {csv_dir}")
    
    csv_files = [f for f in os.listdir(csv_dir) if f.startswith("02_ExperimentResults") and f.endswith(".csv")]
    
    if not csv_files:
        raise FileNotFoundError(f"Nenhum arquivo 02_ExperimentResults_*.csv encontrado em {csv_dir}")
    
    # Ordena por timestamp (mais recente primeiro)
    csv_files.sort(reverse=True)
    latest = os.path.join(csv_dir, csv_files[0])
    logger.info(f"Usando CSV mais recente: {latest}")
    return latest


def load_existing_results(csv_path):
    """
    Carrega os resultados já avaliados do CSV.
    """
    logger.info(f"\n{'='*70}")
    logger.info("ETAPA 1: Carregando resultados já avaliados")
    logger.info(f"{'='*70}")
    logger.info(f"Lendo arquivo: {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"Carregado! {len(df)} linhas encontradas")
    logger.info(f"  Colunas: {list(df.columns)}")
    
    # Agrupar por processo (Folder Name) para contar modelos avaliados
    process_stats = df.groupby('Folder Name').size()
    logger.info(f"\n  Processos encontrados: {len(process_stats)}")
    for process, count in process_stats.items():
        logger.info(f"    - {process}: {count} modelos")
    
    return df


def safe_extract_numeric(value):
    """Extrai um valor numérico de forma segura de qualquer tipo de retorno"""
    if value is None:
        return 0.0
    
    # Se já é float, retorna
    if isinstance(value, float):
        return value
    
    # Se é inteiro, converte
    if isinstance(value, int):
        return float(value)
    
    # Se é dicionário, procura chaves conhecidas recursivamente
    if isinstance(value, dict):
        # Chaves comuns de retorno
        keys_to_try = ['trace_fit_rate', 'fitness', 'Recall', 'recall', 'precision', 'Precision', 
                       'generalization', 'Generalization', 'simplicity', 'Simplicity', 'average', 'value']
        for key in keys_to_try:
            if key in value:
                nested_value = value[key]
                # Se a chave existe, tenta extrair recursivamente
                if isinstance(nested_value, dict):
                    result = safe_extract_numeric(nested_value)
                    if result != 0.0:
                        return result
                elif isinstance(nested_value, (list, tuple)):
                    if len(nested_value) > 0:
                        return safe_extract_numeric(nested_value[0])
                else:
                    try:
                        return float(nested_value) if nested_value is not None else 0.0
                    except (ValueError, TypeError):
                        continue
        # Se nenhuma chave funcionou, retorna 0
        return 0.0
    
    # Se é lista ou tupla, pega o primeiro elemento
    if isinstance(value, (list, tuple)):
        if len(value) > 0:
            return safe_extract_numeric(value[0])
        return 0.0
    
    # Última tentativa: converter string ou número para float
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def calculate_harmonic_mean(recall, precision, generalization, simplicity):
    """
    Calcula a média harmônica das métricas.
    """
    metrics = [recall, precision, generalization, simplicity]
    if all(m > 0 for m in metrics):
        n = len(metrics)
        harmonic_mean = n / sum(1/m for m in metrics)
        return harmonic_mean
    return 0.0

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



def calculate_metric_parallel(metric_name, log, net, initial_marking, final_marking):
    """
    Calcula uma métrica específica de forma isolada para paralelização.
    Retorna (metric_name, value)
    """
    try:
        from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
        from pm4py.algo.evaluation.precision import algorithm as precision_algo
        from pm4py.algo.evaluation.generalization import algorithm as generalization_algo
        from pm4py.algo.evaluation.simplicity import algorithm as simplicity_algo
        from pm4py.algo.filtering.log.cases import case_filter
        
        if metric_name == 'Recall':
            # result = token_replay.apply(log, net, initial_marking, final_marking)

            token_replay_results = token_replay.apply(log, net, initial_marking, final_marking)
            
            # Usar versão otimizada com numpy
            trace_fitnesses = get_trace_fitnesses_optimized(token_replay_results)
            value = compute_fitness_score(trace_fitnesses)
            # value = safe_extract_numeric(result)
        elif metric_name == 'Precision':
            if len(log) > 50000:
                sample_size = 50000
                sampled_log = case_filter.filter_on_ncases(log, sample_size)
                result = precision_algo.apply(sampled_log, net, initial_marking, final_marking)
            else:
                result = precision_algo.apply(log, net, initial_marking, final_marking)            
            # result = precision_algo.apply(log, net, initial_marking, final_marking)
            value = safe_extract_numeric(result)
        elif metric_name == 'Generalization':
            result = generalization_algo.apply(log, net, initial_marking, final_marking)
            value = safe_extract_numeric(result)
        # else:
        #     return (metric_name, 0.0)
        
        # Garantir intervalo [0, 1]
        value = max(0.0, min(1.0, float(value)))
        return (metric_name, value)
    
    except Exception as e:
        logger.warning(f"    Erro ao calcular {metric_name}: {e}")
        return (metric_name, 0.0)


def evaluate_classic_algorithm_models(process_folder, xes_path, output_folder):
    """
    Avalia APENAS os modelos BPMN nas pastas dos algoritmos clássicos.
    
    Procura especificamente por estas pastas:
    - alpha_miner
    - heuristic_miner
    - inductive_miner
    
    Calcula métricas: Recall, Precision, Generalization, Simplicity e Harmonic Mean
    OTIMIZADO: Métricas são calculadas em paralelo usando ThreadPoolExecutor
    """
    from pm4py.objects.log.importer.xes import importer as xes_importer
    from pm4py.objects.bpmn.importer import importer as bpmn_importer
    from pm4py.objects.conversion.bpmn import converter as bpmn_converter
    from pm4py.algo.evaluation.simplicity import algorithm as simplicity_algo
    
    try:
        # Carregar o log XES
        log = xes_importer.apply(xes_path)
        logger.info(f"    Log carregado: {len(log)} traces")
    except Exception as e:
        logger.error(f"    Erro ao carregar XES: {e}")
        return None
    
    results = {}
    
    # Definir os nomes das pastas de algoritmos clássicos a procurar
    algorithm_folders = {
        'alpha_miner': 'Alpha Miner',
        'heuristic_miner': 'Heuristic Miner',
        'inductive_miner': 'Inductive Miner'
    }
    
    # Avaliar cada algoritmo clássico
    for folder_name, algo_label in algorithm_folders.items():
        algo_folder = os.path.join(process_folder, folder_name)
        
        if not os.path.exists(algo_folder):
            logger.warning(f"    Pasta '{folder_name}' nao encontrada")
            continue
        
        logger.info(f"    Processando {algo_label}...")
        logger.info(f"      Iniciando análise de {algo_label}...")
        
        # Procurar por arquivo BPMN nesta pasta
        bpmn_found = False
        for file in os.listdir(algo_folder):
            if file.endswith('.bpmn'):
                try:
                    bpmn_path = os.path.join(algo_folder, file)
                    
                    logger.info(f"      Carregando: {file}")
                    
                    # Importar BPMN uma única vez
                    bpmn_model = bpmn_importer.apply(bpmn_path)
                    
                    # Converter BPMN para Petri net uma única vez
                    net, initial_marking, final_marking = bpmn_converter.apply(bpmn_model)
                    logger.info(f"      Modelo importado com sucesso")
                    
                    # ===== PARALELIZAR CÁLCULO DAS MÉTRICAS =====
                    # Usar ThreadPoolExecutor para calcular Recall, Precision, Generalization em paralelo
                    metrics_dict = {}
                    
                    with ThreadPoolExecutor(max_workers=3) as thread_executor:
                        # Submeter os 3 cálculos em paralelo
                        futures = {
                            thread_executor.submit(
                                calculate_metric_parallel, 
                                metric_name, 
                                log, net, initial_marking, final_marking
                            ): metric_name 
                            for metric_name in ['Recall', 'Precision', 'Generalization']
                        }
                        
                        # Coletar resultados conforme ficam prontos
                        logger.info(f"      Calculando métricas em paralelo...")
                        for future in as_completed(futures):
                            metric_name, value = future.result()
                            metrics_dict[metric_name] = value
                            logger.info(f"        {metric_name}: {value:.4f}")
                    
                    # Calcular Simplicity (não precisa do log)
                    try:
                        simplicity_result = simplicity_algo.apply(net)
                        simplicity = safe_extract_numeric(simplicity_result)
                        simplicity = max(0.0, min(1.0, float(simplicity)))
                        metrics_dict['Simplicity'] = simplicity
                        logger.info(f"        Simplicity: {simplicity:.4f}")
                    except Exception as e:
                        logger.warning(f"      Erro ao calcular Simplicity: {e}")
                        metrics_dict['Simplicity'] = 0.0
                    
                    # Calcular média harmônica APÓS todas as métricas
                    recall = metrics_dict.get('Recall', 0.0)
                    precision = metrics_dict.get('Precision', 0.0)
                    generalization = metrics_dict.get('Generalization', 0.0)
                    simplicity = metrics_dict.get('Simplicity', 0.0)
                    
                    harmonic_mean = calculate_harmonic_mean(recall, precision, generalization, simplicity)
                    metrics_dict['Harmonic_Mean'] = harmonic_mean
                    
                    results[algo_label] = metrics_dict
                    logger.info(f"      Análise de {algo_label} CONCLUÍDA - Média Harmônica: {harmonic_mean:.4f}")
                    bpmn_found = True
                    break  # Encontrou um BPMN, passa para próximo algoritmo
                    
                except Exception as e:
                    logger.warning(f"      Erro ao avaliar {algo_label}: {e}")
                    import traceback
                    logger.warning(traceback.format_exc())
        
        if not bpmn_found:
            logger.warning(f"    Nenhum arquivo BPMN encontrado em '{folder_name}'")
    
    if not results:
        logger.warning(f"    Nenhum modelo de algoritmo classico foi avaliado para {os.path.basename(process_folder)}")
    else:
        logger.info(f"    Total de algoritmos avaliados: {len(results)}")
    
    return results if results else None


def evaluate_process_parallel(args_tuple):
    """
    Função wrapper para avaliar um processo em paralelo.
    Retorna (process_name, result, status)
    
    IMPORTANTE: Esta função roda em um processo SEPARADO (não é thread).
    Cada processo tem sua própria instância do intérprete Python.
    """
    process_name, xes_path, output_folder = args_tuple
    process_folder = os.path.dirname(xes_path)
    
    try:
        logger.info(f"\n>>> INICIANDO: {process_name}")
        logger.info(f"    Caminho: {xes_path}")
        
        # Executar a avaliação completa
        result = evaluate_classic_algorithm_models(process_folder, xes_path, output_folder)
        
        # Marcar como realmente concluído apenas após a execução completa
        if result:
            logger.info(f"<<< CONCLUÍDO: {process_name} - {len(result)} algoritmos avaliados")
            return (process_name, result, 'success')
        else:
            logger.warning(f"<<< CONCLUÍDO (SEM MODELOS): {process_name}")
            return (process_name, None, 'no_models')
            
    except Exception as e:
        logger.error(f"<<< ERRO: {process_name} - {e}")
        traceback.print_exc()
        return (process_name, None, 'error')


def load_classic_algorithms_results(input_folder, xes_files_dict, output_folder, num_workers=None, use_parallel=True):
    """
    Avalia APENAS os modelos dos algoritmos clássicos (Inductive, Heuristic, Alpha).
    Pode executar em paralelo ou sequencial.
    
    Args:
        input_folder: Pasta raiz com os processos
        xes_files_dict: Dicionário {processo: caminho_xes}
        output_folder: Pasta de saída para salvar resultados
        num_workers: Número de workers para paralelização (None = automático)
        use_parallel: Se True, usa ProcessPoolExecutor. Se False, sequencial.
        
    Returns:
        Dicionário com resultados dos algoritmos clássicos
        
    OTIMIZAÇÃO:
    - ProcessPoolExecutor distribui processos entre cores CPU
    - ThreadPoolExecutor dentro de cada processo paraleliza o cálculo de métricas
    - Cada core executa 3 métricas em paralelo (Recall, Precision, Generalization)
    """
    logger.info(f"\n{'='*70}")
    logger.info("ETAPA 2: Avaliando APENAS Modelos dos Algoritmos Classicos")
    logger.info("(Nao vai re-avaliar os modelos BPMN que ja estao no CSV)")
    if use_parallel:
        logger.info(f"Modo: PARALELO OTIMIZADO (2 niveis)")
        logger.info(f"  - Nivel 1: ProcessPoolExecutor distribui processos entre cores CPU")
        logger.info(f"  - Nivel 2: ThreadPoolExecutor paraleliza metricas dentro de cada processo")
    else:
        logger.info(f"Modo: SEQUENCIAL")
    logger.info(f"{'='*70}")
    
    if not xes_files_dict:
        logger.warning("Nenhum arquivo XES encontrado para avaliacao")
        return None
    
    all_results = {}
    
    if not use_parallel:
        # ===== MODO SEQUENCIAL (original) =====
        for process_name, xes_path in xes_files_dict.items():
            logger.info(f"\nProcesso: {process_name}")
            logger.info(f"  XES: {os.path.basename(xes_path)}")
            
            process_folder = os.path.dirname(xes_path)
            
            try:
                result = evaluate_classic_algorithm_models(process_folder, xes_path, output_folder)
                
                if result:
                    all_results[process_name] = result
                    logger.info(f"  Modelos dos algoritmos classicos avaliados com sucesso")
                else:
                    logger.warning(f"  Nenhum modelo de algoritmo classico encontrado para {process_name}")
                    
            except Exception as e:
                logger.warning(f"  Erro ao avaliar {process_name}: {e}")
                traceback.print_exc()
    
    else:
        # ===== MODO PARALELO (2 NÍVEIS) =====
        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)  # Deixar um core livre
        
        logger.info(f"\nConfiguração de Paralelização:")
        logger.info(f"  - Workers (ProcessPoolExecutor): {num_workers}")
        logger.info(f"  - Threads por worker (ThreadPoolExecutor): 3 (Recall, Precision, Generalization)")
        logger.info(f"  - Total de cores disponíveis: {cpu_count()}")
        logger.info(f"  - Total de processos a avaliar: {len(xes_files_dict)}")
        
        # Preparar argumentos para paralelização
        process_args = [
            (process_name, xes_path, output_folder)
            for process_name, xes_path in xes_files_dict.items()
        ]
        
        total = len(process_args)
        logger.info(f"\nIniciando avaliação de {total} processos com {num_workers} workers...\n")
        
        # Executar em paralelo
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submeter todas as tarefas
            futures = {executor.submit(evaluate_process_parallel, args): args[0] for args in process_args}
            
            completed = 0
            processing_status = {}  # Rastrear o status de cada processo
            
            logger.info(f"Tarefas submetidas: {len(futures)}")
            logger.info(f"{'='*70}\n")
            
            # Processar conforme as tarefas terminam
            for future in as_completed(futures):
                completed += 1
                process_name = futures[future]
                
                try:
                    # Desempacotar o novo formato com status
                    result_process_name, result, status = future.result()
                    
                    # Registrar status
                    processing_status[result_process_name] = status
                    
                    if status == 'success' and result:
                        all_results[result_process_name] = result
                        logger.info(f"[{completed:2d}/{total}] {result_process_name:30s} | SUCESSO")
                        
                    elif status == 'no_models':
                        logger.warning(f"[{completed:2d}/{total}] {result_process_name:30s} | SEM MODELOS")
                        
                    elif status == 'error':
                        logger.error(f"[{completed:2d}/{total}] {result_process_name:30s} | ERRO")
                        
                except Exception as e:
                    logger.error(f"[{completed:2d}/{total}] {process_name:30s} | ERRO AO PROCESSAR - {e}")
                    traceback.print_exc()
                    processing_status[process_name] = 'error'
            
            # Aguardar explicitamente que todos os workers finalizem
            executor.shutdown(wait=True)
            
            logger.info(f"\n{'='*70}")
            logger.info(f"CONCLUIDO: Todas as {completed} tarefas finalizaram")
            logger.info(f"{'='*70}\n")
            
            # Resumo de status
            logger.info("Resumo de Status:")
            success_count = sum(1 for s in processing_status.values() if s == 'success')
            error_count = sum(1 for s in processing_status.values() if s == 'error')
            no_models_count = sum(1 for s in processing_status.values() if s == 'no_models')
            logger.info(f"  - Sucesso: {success_count}")
            logger.info(f"  - Erro: {error_count}")
            logger.info(f"  - Sem modelos: {no_models_count}")
            logger.info(f"  - Total: {completed}/{total}\n")
    
    logger.info(f"Algoritmos classicos avaliados para {len(all_results)} processos")
    return all_results if all_results else None




def merge_and_compare_results(your_results_df, classic_results, output_folder):
    """
    Compara os resultados dos seus modelos com os dos algoritmos clássicos.
    Gera estatísticas e gráficos POR PROCESSO.
    """
    logger.info(f"\n{'='*70}")
    logger.info("ETAPA 3: Comparacao e Visualizacao dos Resultados POR PROCESSO")
    logger.info(f"{'='*70}")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Métricas a comparar
        metrics = ['Recall', 'Precision', 'Generalization', 'Simplicity', 'Harmonic_Mean']
        
        # Converter resultados dos algoritmos clássicos para DataFrame
        classic_data = []
        if classic_results:
            for process_name, algo_results in classic_results.items():
                if algo_results and isinstance(algo_results, dict):
                    for algo_name, metrics_dict in algo_results.items():
                        if isinstance(metrics_dict, dict):
                            row = {'Process': process_name, 'Algorithm': algo_name}
                            row.update(metrics_dict)
                            classic_data.append(row)
        
        if not classic_data:
            logger.warning("Nenhum dado de algoritmos classicos para comparar")
            return False
            
        classic_df = pd.DataFrame(classic_data)
        
        # Identificar processos comuns entre seus resultados e algoritmos clássicos
        # Seus resultados estão no formato de estatísticas agregadas por processo
        # Precisamos identificar os processos no CSV de métricas
        
        # Assumindo que o CSV tem uma coluna 'Metric' indicando qual métrica
        # e colunas para cada processo
        processes_in_classic = classic_df['Process'].unique()
        logger.info(f"\nProcessos encontrados nos algoritmos classicos: {len(processes_in_classic)}")
        for proc in sorted(processes_in_classic):
            logger.info(f"  - {proc}")
        
        # Gerar gráficos individuais por processo
        generate_per_process_comparison_plots(your_results_df, classic_df, output_folder, processes_in_classic)
        
        # Gerar comparação de média harmônica por processo
        generate_per_process_harmonic_mean_comparison(your_results_df, classic_df, output_folder, processes_in_classic)
        
        # Salvar tabela de resumo por processo
        save_per_process_summary(your_results_df, classic_df, output_folder, processes_in_classic)
        
        return True
    
    except Exception as e:
        logger.warning(f"Erro ao gerar visualizacoes: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_per_process_comparison_plots(your_df, classic_df, output_folder, processes):
    """
    Gera gráficos comparativos POR PROCESSO com as 4 métricas + média harmônica.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    metrics = ['Recall', 'Precision', 'Generalization', 'Simplicity']
    
    # Criar pasta para gráficos por processo
    process_plots_folder = os.path.join(output_folder, "GRAFICOS_POR_PROCESSO")
    if not os.path.exists(process_plots_folder):
        os.makedirs(process_plots_folder)
    
    logger.info(f"\nGerando graficos individuais para {len(processes)} processos...")
    
    for process in sorted(processes):
        try:
            logger.info(f"  - Gerando grafico para: {process}")
            
            # Filtrar dados do processo
            classic_process_data = classic_df[classic_df['Process'] == process]
            
            if len(classic_process_data) == 0:
                logger.warning(f"    Sem dados classicos para {process}")
                continue
            
            # Extrair média dos seus modelos para este processo do CSV de estatísticas
            # O CSV tem formato: cada linha é um processo, colunas são as métricas
            your_process_metrics = {}
            
            # Procurar o processo no DataFrame
            process_row = your_df[your_df['Folder Name'] == process]
            
            if process_row.empty:
                logger.warning(f"    Sem dados dos seus modelos para {process}")
                continue
            
            # Extrair as médias das métricas
            for metric in metrics:
                metric_col = f'{metric} Mean'
                if metric_col in your_df.columns:
                    value = process_row[metric_col].values[0]
                    if pd.notna(value):
                        your_process_metrics[metric] = value
            
            if not your_process_metrics:
                logger.warning(f"    Sem metricas validas para {process}")
                continue
            
            # Criar figura
            sns.set_style("whitegrid")
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f"Processo: {process}\nComparacao: Seus Experimentos vs Algoritmos Classicos", 
                        fontsize=14, fontweight='bold')
            
            axes = axes.flatten()
            
            for idx, metric in enumerate(metrics):
                ax = axes[idx]
                
                if metric not in classic_process_data.columns:
                    ax.text(0.5, 0.5, f'{metric} nao disponivel', ha='center', va='center')
                    continue
                
                # Seus dados (média do processo)
                your_value = your_process_metrics.get(metric, None)
                
                # Dados dos algoritmos clássicos
                classic_values = classic_process_data[metric].dropna().values
                
                if your_value is None or len(classic_values) == 0:
                    ax.text(0.5, 0.5, f'Sem dados para {metric}', ha='center', va='center')
                    continue
                
                # Preparar dados para o gráfico de barras
                algorithms = ['Experimentos SBMN\n(Média)'] + classic_process_data['Algorithm'].tolist()
                values = [your_value] + classic_values.tolist()
                
                # Cores: azul para experimentos SBMN, cores diferentes para cada algoritmo clássico
                classic_colors = ['#e74c3c', '#f39c12', '#27ae60']  # Vermelho, Laranja, Verde
                colors = ['#3498db'] + classic_colors[:len(classic_values)]
                
                # Criar gráfico de barras
                bars = ax.bar(range(len(algorithms)), values, color=colors, alpha=0.7, edgecolor='black')
                
                ax.set_ylabel(metric, fontsize=11, fontweight='bold')
                ax.set_ylim([0, 1.05])
                ax.set_xticks(range(len(algorithms)))
                ax.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=9)
                ax.grid(True, alpha=0.3, axis='y')
                
                # Adicionar valores nas barras
                for i, (bar, val) in enumerate(zip(bars, values)):
                    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                           f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
            
            plt.tight_layout()
            safe_process_name = process.replace('/', '_').replace('\\', '_')
            output_path = os.path.join(process_plots_folder, f"{safe_process_name}_metricas.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"    Salvo: {output_path}")
            plt.close()
            
        except Exception as e:
            logger.warning(f"    Erro ao gerar grafico para {process}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\nGraficos por processo salvos em: {process_plots_folder}")


def generate_per_process_harmonic_mean_comparison(your_df, classic_df, output_folder, processes):
    """
    Gera gráficos específicos da média harmônica POR PROCESSO.
    """
def generate_per_process_harmonic_mean_comparison(your_df, classic_df, output_folder, processes):
    """
    Gera gráficos específicos da média harmônica POR PROCESSO.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Criar pasta para gráficos de média harmônica
    harmonic_plots_folder = os.path.join(output_folder, "GRAFICOS_MEDIA_HARMONICA")
    if not os.path.exists(harmonic_plots_folder):
        os.makedirs(harmonic_plots_folder)
    
    logger.info(f"\nGerando graficos de media harmonica para {len(processes)} processos...")
    
    for process in sorted(processes):
        try:
            logger.info(f"  - Gerando grafico de media harmonica para: {process}")
            
            # Filtrar dados do processo
            classic_process_data = classic_df[classic_df['Process'] == process]
            
            if len(classic_process_data) == 0:
                logger.warning(f"    Sem dados classicos para {process}")
                continue
            
            if 'Harmonic_Mean' not in classic_process_data.columns:
                logger.warning(f"    Harmonic_Mean nao disponivel para {process}")
                continue
            
            # Extrair média harmônica dos seus modelos para este processo
            # Procurar o processo no DataFrame
            process_row = your_df[your_df['Folder Name'] == process]
            
            if process_row.empty:
                logger.warning(f"    Processo {process} nao encontrado nos seus dados")
                continue
            
            # Calcular média harmônica a partir das métricas disponíveis
            your_harmonic = None
            if 'Harmonic_Mean Mean' in your_df.columns:
                your_harmonic = process_row['Harmonic_Mean Mean'].values[0]
            else:
                # Calcular média harmônica manualmente se não existir
                recall = process_row['Recall Mean'].values[0] if 'Recall Mean' in your_df.columns else None
                precision = process_row['Precision Mean'].values[0] if 'Precision Mean' in your_df.columns else None
                generalization = process_row['Generalization Mean'].values[0] if 'Generalization Mean' in your_df.columns else None
                simplicity = process_row['Simplicity Mean'].values[0] if 'Simplicity Mean' in your_df.columns else None
                
                if all(pd.notna(v) and v > 0 for v in [recall, precision, generalization, simplicity]):
                    your_harmonic = calculate_harmonic_mean(recall, precision, generalization, simplicity)
            
            if your_harmonic is None or pd.isna(your_harmonic):
                logger.warning(f"    Sem media harmonica dos seus modelos para {process}")
                continue
            
            # Dados dos algoritmos clássicos
            classic_harmonics = classic_process_data['Harmonic_Mean'].dropna().values
            
            if len(classic_harmonics) == 0:
                logger.warning(f"    Sem dados de media harmonica dos algoritmos classicos para {process}")
                continue
            
            # Criar gráfico
            sns.set_style("whitegrid")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Preparar dados para o gráfico de barras
            algorithms = ['Experimentos SBMN\n(Média)'] + classic_process_data['Algorithm'].tolist()
            values = [your_harmonic] + classic_harmonics.tolist()
            
            # Cores: azul para experimentos SBMN, cores diferentes para cada algoritmo clássico
            classic_colors = ['#e74c3c', '#f39c12', '#27ae60']  # Vermelho, Laranja, Verde
            colors = ['#3498db'] + classic_colors[:len(classic_harmonics)]
            
            # Criar gráfico de barras
            bars = ax.bar(range(len(algorithms)), values, color=colors, alpha=0.7, edgecolor='black')
            
            ax.set_ylabel('Media Harmonica', fontsize=12, fontweight='bold')
            ax.set_title(f'Processo: {process}\nComparacao da Media Harmonica', fontsize=14, fontweight='bold')
            ax.set_ylim([0, 1.05])
            ax.set_xticks(range(len(algorithms)))
            ax.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Adicionar valores nas barras
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            plt.tight_layout()
            safe_process_name = process.replace('/', '_').replace('\\', '_')
            output_path = os.path.join(harmonic_plots_folder, f"{safe_process_name}_harmonica.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"    Salvo: {output_path}")
            plt.close()
            
        except Exception as e:
            logger.warning(f"    Erro ao gerar grafico de media harmonica para {process}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\nGraficos de media harmonica salvos em: {harmonic_plots_folder}")


def save_per_process_summary(your_df, classic_df, output_folder, processes):
    """
    Salva tabela de resumo com comparação por processo.
    Inclui tabela específica com Média Harmônica por processo e algoritmo.
    """
    try:
        logger.info(f"\nGerando resumo comparativo por processo...")
        
        metrics = ['Recall', 'Precision', 'Generalization', 'Simplicity', 'Harmonic_Mean']
        summary_rows = []
        harmonic_rows = []
        
        for process in sorted(processes):
            classic_process_data = classic_df[classic_df['Process'] == process]
            
            if len(classic_process_data) == 0:
                continue
            
            row = {'Processo': process}
            harmonic_row = {'Processo': process}
            
            # Procurar o processo nos seus dados
            process_row = your_df[your_df['Folder Name'] == process]
            
            for metric in metrics:
                # Seus dados
                if not process_row.empty:
                    metric_col = f'{metric} Mean'
                    if metric_col in your_df.columns:
                        value = process_row[metric_col].values[0]
                        if pd.notna(value):
                            row[f'{metric}_SeuModelo'] = value
                    
                    # Adicionar std também se disponível
                    std_col = f'{metric} Std Dev'
                    if std_col in your_df.columns:
                        std_value = process_row[std_col].values[0]
                        if pd.notna(std_value):
                            row[f'{metric}_SeuModelo_Std'] = std_value
                
                # Dados clássicos (média dos algoritmos)
                if metric in classic_process_data.columns:
                    classic_values = classic_process_data[metric].dropna()
                    if len(classic_values) > 0:
                        row[f'{metric}_Classicos_Media'] = classic_values.mean()
                        row[f'{metric}_Classicos_Std'] = classic_values.std()
            
            summary_rows.append(row)
            
            # ===== TABELA SEPARADA COM MÉDIA HARMÔNICA =====
            # Calcular Média Harmônica dos experimentos SBMN a partir das 4 métricas
            your_harmonic = None
            
            if not process_row.empty:
                recall_val = process_row.get('Recall Mean').values[0] if 'Recall Mean' in your_df.columns else None
                precision_val = process_row.get('Precision Mean').values[0] if 'Precision Mean' in your_df.columns else None
                generalization_val = process_row.get('Generalization Mean').values[0] if 'Generalization Mean' in your_df.columns else None
                simplicity_val = process_row.get('Simplicity Mean').values[0] if 'Simplicity Mean' in your_df.columns else None
                
                logger.debug(f"Processo {process}: Recall={recall_val}, Precision={precision_val}, Gen={generalization_val}, Simp={simplicity_val}")
                
                # Calcular harmônica apenas se todas as métricas existem e são válidas
                if all(pd.notna(v) and v > 0 for v in [recall_val, precision_val, generalization_val, simplicity_val]):
                    your_harmonic = calculate_harmonic_mean(recall_val, precision_val, generalization_val, simplicity_val)
                    logger.debug(f"Média Harmônica calculada para {process}: {your_harmonic:.4f}")
            
            if your_harmonic is not None and pd.notna(your_harmonic):
                harmonic_row['Experimentos_SBMN'] = round(float(your_harmonic), 4)
            else:
                harmonic_row['Experimentos_SBMN'] = '-'
            
            # Média harmônica por algoritmo clássico
            for algo in sorted(classic_process_data['Algorithm'].unique()):
                algo_data = classic_process_data[classic_process_data['Algorithm'] == algo]
                if 'Harmonic_Mean' in algo_data.columns:
                    harmonic_values = algo_data['Harmonic_Mean'].dropna()
                    if len(harmonic_values) > 0:
                        harmonic_row[algo] = round(harmonic_values.mean(), 4)
                    else:
                        harmonic_row[algo] = '-'
                else:
                    harmonic_row[algo] = '-'
            
            harmonic_rows.append(harmonic_row)
        
        # Salvar tabela completa de resumo
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_path = os.path.join(output_folder, "RESUMO_COMPARACAO_POR_PROCESSO.csv")
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Resumo comparativo completo salvo em: {summary_path}")
        
        # Salvar tabela de Média Harmônica (tabela limpa e clara)
        if harmonic_rows:
            harmonic_df = pd.DataFrame(harmonic_rows)
            # Reordenar colunas: Processo primeiro, depois Experimentos_SBMN, depois algoritmos
            cols = ['Processo', 'Experimentos_SBMN'] + [c for c in harmonic_df.columns if c not in ['Processo', 'Experimentos_SBMN']]
            harmonic_df = harmonic_df[cols]
            
            harmonic_path = os.path.join(output_folder, "MEDIA_HARMONICA_POR_PROCESSO.csv")
            harmonic_df.to_csv(harmonic_path, index=False)
            logger.info(f"Tabela de Média Harmônica por processo salva em: {harmonic_path}")
            
            # Formatação melhorada da tabela com alinhamento
            logger.info(f"\n{'='*100}")
            logger.info(f"TABELA: Média Harmônica por Processo")
            logger.info(f"{'='*100}")
            
            # Usar formatação com pandas to_string com espaçamento melhorado
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)
            
            # Formatar com alinhamento
            formatted_table = harmonic_df.to_string(index=False)
            for line in formatted_table.split('\n'):
                logger.info(line)
            logger.info(f"{'='*100}\n")
        
    except Exception as e:
        logger.warning(f"Erro ao gerar resumo: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Compara CSVs gerados com resultados dos algoritmos classicos ja existentes")
    parser.add_argument("--csv", type=str, default=None, help="Caminho para o CSV com resultados ja avaliados (se nao especificado, usa o mais recente da pasta --csv-dir)")
    parser.add_argument("--csv-dir", type=str, default="OUTPUT/OUTPUT_EXPERIMENTOS_COMRM", help="Pasta onde estao seus CSVs ja gerados (padrao: OUTPUT)")
    parser.add_argument("--input", type=str, default="INPUT/EXPERIMENTOS_COMRM", help="Pasta de entrada com os processos (contem as pastas de algoritmos classicos)")
    parser.add_argument("--output", type=str, default="OUTPUT/QUICK_COMPARISON", help="Pasta de saida para resultados da comparacao")
    parser.add_argument("--parallel", action="store_true", default=True, help="Usar paralelizacao (padrao: True)")
    parser.add_argument("--no-parallel", dest="parallel", action="store_false", help="Desabilitar paralelizacao (executa em sequencia)")
    parser.add_argument("--workers", type=int, default=None, help="Numero de workers para paralelizacao (padrao: auto - usa cpu_count - 1)")
    
    args = parser.parse_args()
    
    # Buscar o CSV de métricas (03_MetricStatistics_*.csv) ao invés do de resultados
    metric_csv_path = get_latest_metric_csv(args.csv_dir)
    
    logger.info("\n" + "="*70)
    logger.info("COMPARACAO: Modelos BPMN vs Algoritmos Classicos")
    logger.info("="*70 + "\n")
    
    # ETAPA 1: Carregar CSV de métricas dos seus modelos
    logger.info(f"\nETAPA 1: Carregando metricas dos seus modelos BPMN")
    logger.info(f"Lendo: {metric_csv_path}")
    
    your_results = pd.read_csv(metric_csv_path)
    logger.info(f"Carregado! {len(your_results)} linhas encontradas")
    
    # ETAPA 2: Encontrar arquivos XES
    logger.info(f"\n{'='*70}")
    logger.info("Procurando arquivos XES nos processos")
    logger.info(f"{'='*70}")
    xes_files_dict = find_xes_files(args.input)
    
    if not xes_files_dict:
        logger.error(f"Nenhum arquivo XES encontrado em {args.input}")
        logger.info("  Verifique se a pasta de entrada existe e contem arquivos .xes")
        sys.exit(1)
    
    # ETAPA 3: Avaliar modelos dos algoritmos clássicos com XES
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    classic_results = load_classic_algorithms_results(
        args.input, 
        xes_files_dict, 
        args.output,
        num_workers=args.workers,
        use_parallel=args.parallel
    )
    
    if not classic_results:
        logger.warning("Nenhum resultado de algoritmo classico foi gerado.")
        classic_results = None
    
    # ETAPA 4: Comparar e visualizar
    merge_and_compare_results(your_results, classic_results, args.output)
    
    logger.info("\n" + "="*70)
    logger.info("PROCESSAMENTO CONCLUIDO!")
    logger.info(f"Resultados salvos em: {args.output}")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nExecucao interrompida pelo usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
