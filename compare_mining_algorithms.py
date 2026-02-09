"""
Módulo para comparar métricas dos modelos gerados com algoritmos clássicos de mineração.
Gera gráficos comparativos de:
1. Média das métricas (Recall, Precision, Generalization, Simplicity)
2. Média Harmônica das métricas
"""

import os
import pm4py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from pathlib import Path
from pm4py.objects.conversion.bpmn import converter as bpmn_converter
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.evaluation.precision import algorithm as precision_algo
from pm4py.algo.evaluation.generalization import algorithm as generalization_algo
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_algo
import gc
import time

logger = logging.getLogger(__name__)

def evaluate_model_for_comparison(log, petri_net, initial_marking, final_marking):
    """
    Avalia um modelo Petri Net com as métricas básicas.
    Retorna: (recall, precision, generalization, simplicity)
    """
    try:
        # Token replay para fitness (recall)
        token_replay_results = token_replay.apply(
            log, petri_net, initial_marking, final_marking,
            variant=token_replay.Variants.TOKEN_REPLAY
        )
        
        trace_fitnesses = np.array([res.get("trace_fitness", 0.0) 
                                   for res in token_replay_results], dtype=np.float32)
        recall = float(np.mean(trace_fitnesses)) if len(trace_fitnesses) > 0 else 0.0
        
        # Outras métricas
        precision = precision_algo.apply(log, petri_net, initial_marking, final_marking)
        generalization = generalization_algo.apply(log, petri_net, initial_marking, final_marking)
        simplicity = simplicity_algo.apply(petri_net)
        
        return recall, precision, generalization, simplicity
        
    except Exception as e:
        logger.error(f"Erro ao avaliar modelo: {e}")
        return 0.0, 0.0, 0.0, 0.0


def evaluate_bpmn_file(bpmn_path, log):
    """
    Carrega um arquivo BPMN e o avalia.
    Retorna um dicionário com as métricas.
    """
    try:
        # Carregar BPMN
        bpmn_graph = pm4py.read_bpmn(bpmn_path)
        petri_net, initial_marking, final_marking = bpmn_converter.apply(
            bpmn_graph, variant=bpmn_converter.Variants.TO_PETRI_NET
        )
        
        # Avaliar
        recall, precision, generalization, simplicity = evaluate_model_for_comparison(
            log, petri_net, initial_marking, final_marking
        )
        
        # Limpar memória
        del bpmn_graph, petri_net, initial_marking, final_marking
        gc.collect()
        
        return {
            "recall": recall,
            "precision": precision,
            "generalization": generalization,
            "simplicity": simplicity
        }
        
    except Exception as e:
        logger.warning(f"Erro ao processar {bpmn_path}: {e}")
        return None


def calculate_harmonic_mean(metrics_dict):
    """
    Calcula a média harmônica de um conjunto de métricas.
    metrics_dict: dicionário com as 4 métricas
    """
    values = [metrics_dict["recall"], metrics_dict["precision"], 
              metrics_dict["generalization"], metrics_dict["simplicity"]]
    
    # Evitar divisão por zero
    if any(v <= 0 for v in values):
        return 0.0
    
    n = len(values)
    harmonic_mean = n / sum(1/v for v in values)
    return harmonic_mean


def get_generated_models_metrics(models_folder, log):
    """
    Avalia todos os modelos gerados (S0_*, S1_*, etc.) em uma pasta.
    Retorna DataFrame com as métricas.
    """
    models_data = []
    
    # Encontrar todos os arquivos de solução
    solution_files = sorted([f for f in os.listdir(models_folder) 
                            if f.startswith("S") and f.endswith(".bpmn")])
    
    if not solution_files:
        logger.warning(f"Nenhum modelo gerado encontrado em {models_folder}")
        return pd.DataFrame()
    
    logger.info(f"  Avaliando {len(solution_files)} modelos gerados...")
    
    for i, model_file in enumerate(solution_files, 1):
        model_path = os.path.join(models_folder, model_file)
        
        metrics = evaluate_bpmn_file(model_path, log)
        if metrics is not None:
            harmonic_mean = calculate_harmonic_mean(metrics)
            
            models_data.append({
                "model": model_file,
                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "generalization": metrics["generalization"],
                "simplicity": metrics["simplicity"],
                "harmonic_mean": harmonic_mean
            })
            
            if i % 10 == 0:
                logger.info(f"    Processados {i}/{len(solution_files)} modelos...")
    
    df = pd.DataFrame(models_data)
    logger.info(f"  ✓ {len(df)} modelos avaliados com sucesso")
    
    return df


def get_classic_algorithms_metrics(algorithms_folder, log):
    """
    Avalia os modelos dos algoritmos clássicos (inductive_miner, alpha_miner, heuristic_miner).
    Retorna dicionário com as métricas para cada algoritmo.
    """
    algorithms = ["inductive_miner", "alpha_miner", "heuristic_miner"]
    algorithms_metrics = {}
    
    for algo in algorithms:
        algo_path = os.path.join(algorithms_folder, algo)
        
        if not os.path.exists(algo_path):
            logger.warning(f"Pasta do algoritmo não encontrada: {algo_path}")
            continue
        
        # Encontrar o arquivo BPMN na pasta do algoritmo
        bpmn_files = [f for f in os.listdir(algo_path) if f.endswith(".bpmn")]
        
        if not bpmn_files:
            logger.warning(f"Nenhum arquivo BPMN encontrado em {algo_path}")
            continue
        
        bpmn_file = bpmn_files[0]  # Usar o primeiro arquivo BPMN encontrado
        bpmn_path = os.path.join(algo_path, bpmn_file)
        
        logger.info(f"  Avaliando {algo}...")
        
        metrics = evaluate_bpmn_file(bpmn_path, log)
        if metrics is not None:
            harmonic_mean = calculate_harmonic_mean(metrics)
            
            algorithms_metrics[algo] = {
                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "generalization": metrics["generalization"],
                "simplicity": metrics["simplicity"],
                "harmonic_mean": harmonic_mean
            }
            logger.info(f"    ✓ {algo} avaliado com sucesso")
        else:
            logger.warning(f"    ✗ Erro ao avaliar {algo}")
    
    return algorithms_metrics


def plot_metrics_comparison(process_name, generated_df, algorithms_metrics, output_dir):
    """
    Cria gráfico comparativo de médias das métricas.
    
    Args:
        process_name: Nome do processo
        generated_df: DataFrame com métricas dos modelos gerados
        algorithms_metrics: Dicionário com métricas dos algoritmos clássicos
        output_dir: Diretório para salvar o gráfico
    """
    metrics = ["recall", "precision", "generalization", "simplicity"]
    
    # Calcular médias dos modelos gerados
    generated_means = {
        metric: generated_df[metric].mean() for metric in metrics
    }
    
    # Preparar dados para o gráfico
    plot_data = {"Generated Models (Mean)": generated_means}
    for algo, metrics_dict in algorithms_metrics.items():
        plot_data[algo] = {metric: metrics_dict[metric] for metric in metrics}
    
    # Criar figura
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Métricas de Qualidade - {process_name}\nComparação de Algoritmos", 
                 fontsize=16, fontweight='bold')
    
    axes_flat = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes_flat[idx]
        
        # Preparar dados para este métrica
        comparison_data = []
        labels = []
        
        for name, metrics_dict in plot_data.items():
            comparison_data.append(metrics_dict[metric])
            labels.append(name)
        
        # Cores: azul para modelos gerados, verde/laranja/vermelho para algoritmos
        colors = ['#1f77b4'] + ['#2ca02c', '#ff7f0e', '#d62728'][:len(algorithms_metrics)]
        
        # Criar gráfico de barras
        bars = ax.bar(range(len(labels)), comparison_data, color=colors[:len(labels)], alpha=0.8)
        
        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title(metric.capitalize(), fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    # Salvar gráfico
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"01_MetricasComparativas_{process_name}_{timestamp}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"  ✓ Gráfico de métricas salvo: {filename}")
    plt.close()


def plot_harmonic_mean_comparison(process_name, generated_df, algorithms_metrics, output_dir):
    """
    Cria gráfico comparativo de média harmônica.
    
    Args:
        process_name: Nome do processo
        generated_df: DataFrame com métricas dos modelos gerados
        algorithms_metrics: Dicionário com métricas dos algoritmos clássicos
        output_dir: Diretório para salvar o gráfico
    """
    # Calcular média harmônica dos modelos gerados
    generated_harmonic_mean = generated_df["harmonic_mean"].mean()
    
    # Preparar dados para o gráfico
    plot_data = {
        "Generated Models (Mean)": generated_harmonic_mean
    }
    
    for algo, metrics_dict in algorithms_metrics.items():
        plot_data[algo] = metrics_dict["harmonic_mean"]
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = list(plot_data.keys())
    values = list(plot_data.values())
    colors = ['#1f77b4'] + ['#2ca02c', '#ff7f0e', '#d62728'][:len(algorithms_metrics)]
    
    # Criar gráfico de barras
    bars = ax.bar(labels, values, color=colors[:len(labels)], alpha=0.8)
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Média Harmônica', fontsize=12, fontweight='bold')
    ax.set_title(f"Média Harmônica das Métricas - {process_name}\nComparação de Algoritmos", 
                fontsize=14, fontweight='bold')
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(values) * 1.2])
    
    plt.tight_layout()
    
    # Salvar gráfico
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"02_MediaHarmonicaComparativa_{process_name}_{timestamp}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"  ✓ Gráfico de média harmônica salvo: {filename}")
    plt.close()


def create_comparison_summary_table(process_name, generated_df, algorithms_metrics, output_dir):
    """
    Cria uma tabela resumida com as comparações.
    
    Args:
        process_name: Nome do processo
        generated_df: DataFrame com métricas dos modelos gerados
        algorithms_metrics: Dicionário com métricas dos algoritmos clássicos
        output_dir: Diretório para salvar a tabela
    """
    # Calcular estatísticas dos modelos gerados
    summary_data = {
        "Algorithm/Metric": ["Generated Models (Mean)", "Generated Models (Std)"],
        "Recall": [
            f"{generated_df['recall'].mean():.4f}",
            f"{generated_df['recall'].std():.4f}"
        ],
        "Precision": [
            f"{generated_df['precision'].mean():.4f}",
            f"{generated_df['precision'].std():.4f}"
        ],
        "Generalization": [
            f"{generated_df['generalization'].mean():.4f}",
            f"{generated_df['generalization'].std():.4f}"
        ],
        "Simplicity": [
            f"{generated_df['simplicity'].mean():.4f}",
            f"{generated_df['simplicity'].std():.4f}"
        ],
        "Harmonic Mean": [
            f"{generated_df['harmonic_mean'].mean():.4f}",
            f"{generated_df['harmonic_mean'].std():.4f}"
        ]
    }
    
    # Adicionar algoritmos clássicos
    for algo, metrics_dict in algorithms_metrics.items():
        summary_data["Algorithm/Metric"].append(algo)
        summary_data["Recall"].append(f"{metrics_dict['recall']:.4f}")
        summary_data["Precision"].append(f"{metrics_dict['precision']:.4f}")
        summary_data["Generalization"].append(f"{metrics_dict['generalization']:.4f}")
        summary_data["Simplicity"].append(f"{metrics_dict['simplicity']:.4f}")
        summary_data["Harmonic Mean"].append(f"{metrics_dict['harmonic_mean']:.4f}")
    
    summary_df = pd.DataFrame(summary_data)
    
    # Salvar como CSV
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(output_dir, f"03_SummaryTable_{process_name}_{timestamp}.csv")
    summary_df.to_csv(csv_filename, index=False)
    logger.info(f"  ✓ Tabela de resumo salva: {csv_filename}")
    
    return summary_df


def compare_process_algorithms(process_folder_path, xes_log_path, output_dir):
    """
    Função principal que compara os modelos gerados com os algoritmos clássicos.
    
    Args:
        process_folder_path: Caminho da pasta do processo (contém S*.bpmn e pastas de algoritmos)
        xes_log_path: Caminho do arquivo XES para avaliação
        output_dir: Diretório para salvar os gráficos
    """
    try:
        process_name = os.path.basename(process_folder_path)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Comparando Algoritmos para: {process_name}")
        logger.info(f"{'='*70}")
        
        # Carregar log
        logger.info("Carregando log XES...")
        log = pm4py.read_xes(xes_log_path)
        logger.info(f"  ✓ Log carregado com {len(log)} traces")
        
        # Avaliar modelos gerados
        logger.info("\nAvaliando modelos gerados...")
        generated_df = get_generated_models_metrics(process_folder_path, log)
        
        if generated_df.empty:
            logger.warning(f"Nenhum modelo gerado foi avaliado para {process_name}")
            return
        
        # Avaliar algoritmos clássicos
        logger.info("\nAvaliando algoritmos clássicos...")
        algorithms_metrics = get_classic_algorithms_metrics(process_folder_path, log)
        
        if not algorithms_metrics:
            logger.warning(f"Nenhum algoritmo clássico encontrado para {process_name}")
            return
        
        # Criar gráficos
        logger.info("\nGerando gráficos...")
        
        # Gráfico 1: Comparativo de métricas
        plot_metrics_comparison(process_name, generated_df, algorithms_metrics, output_dir)
        
        # Gráfico 2: Comparativo de média harmônica
        plot_harmonic_mean_comparison(process_name, generated_df, algorithms_metrics, output_dir)
        
        # Tabela resumida
        summary_df = create_comparison_summary_table(process_name, generated_df, algorithms_metrics, output_dir)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✓ Comparação concluída com sucesso para {process_name}")
        logger.info(f"{'='*70}\n")
        
        return {
            "process": process_name,
            "generated_models": len(generated_df),
            "algorithms_compared": len(algorithms_metrics),
            "summary": summary_df
        }
        
    except Exception as e:
        logger.error(f"Erro ao comparar algoritmos para {process_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def batch_compare_all_processes(main_input_folder, xes_log_filename, output_dir):
    """
    Processa múltiplos processos e gera comparações para todos.
    
    Args:
        main_input_folder: Pasta principal com os processos (ex: INPUT/EXPERIMENTOS_SEMRM)
        xes_log_filename: Nome do arquivo XES (ex: BPIC14-PreProcessed-Filtered.xes)
        output_dir: Diretório para salvar os gráficos
    """
    results = []
    
    # Encontrar todas as pastas de processos
    process_folders = [
        d for d in os.listdir(main_input_folder)
        if os.path.isdir(os.path.join(main_input_folder, d)) 
        and not d.startswith(".")
    ]
    
    logger.info(f"\nEncontrados {len(process_folders)} processos para comparar")
    
    for i, process_folder in enumerate(sorted(process_folders), 1):
        process_path = os.path.join(main_input_folder, process_folder)
        xes_log_path = os.path.join(process_path, xes_log_filename)
        
        # Verificar se o arquivo XES existe
        if not os.path.exists(xes_log_path):
            logger.warning(f"[{i}/{len(process_folders)}] Arquivo XES não encontrado em {process_folder}")
            continue
        
        logger.info(f"\n[{i}/{len(process_folders)}] Processando: {process_folder}")
        
        result = compare_process_algorithms(process_path, xes_log_path, output_dir)
        if result is not None:
            results.append(result)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Processamento em lote concluído!")
    logger.info(f"Total de processos comparados: {len(results)}")
    logger.info(f"{'='*70}\n")
    
    return results


# if __name__ == "__main__":
#     # Exemplo de uso
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s'
#     )
    
#     # Processar um único processo
#     # process_folder = "INPUT/EXPERIMENTOS_SEMRM/BPIC14_ITIL_FILTERED-Pruned"
#     # xes_log = "INPUT/EXPERIMENTOS_SEMRM/BPIC14_ITIL_FILTERED-Pruned/BPIC14-PreProcessed-Filtered.xes"
#     # output_folder = "OUTPUT/SEMRM"
    
#     if os.path.exists(process_folder) and os.path.exists(xes_log):
#         compare_process_algorithms(process_folder, xes_log, output_folder)
#     else:
#         print(f"Pasta ou arquivo XES não encontrado")
