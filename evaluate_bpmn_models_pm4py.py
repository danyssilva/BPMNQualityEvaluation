import os
import pm4py
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

def evaluate_model(log, petri_net, initial_marking, final_marking):
    token_replay_results = token_replay.apply(log, petri_net, initial_marking, final_marking)
    trace_fitnesses = [res["trace_fitness"] for res in token_replay_results]
    recall = sum(trace_fitnesses) / len(trace_fitnesses) if trace_fitnesses else 0

    precision = precision_algo.apply(log, petri_net, initial_marking, final_marking)
    generalization = generalization_algo.apply(log, petri_net, initial_marking, final_marking)
    simplicity = simplicity_algo.apply(petri_net)

    return recall, precision, generalization, simplicity

def plot_results_per_metric(results, folder_name):
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
        # ax.plot(range(len(values[metric])), values[metric], marker='o', linestyle='-', label=metric, color=colors[i])
        print(f"Total modelos: {range(1, len(models)+1)}")
        ax.plot(range(1, len(models)+1), values[metric], marker='o', linestyle='-', label=metric, color=colors[i])
        ax.set_title(f'{metric}', fontsize=10)
        # ax.set_xlabel('Model', labelpad=0.05)
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
    plt.show()

def plot_box_plots(global_results, folder_solution_counts):
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
    plt.show()

def plot_execution_times(global_execution_times, folder_solution_counts):
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
    plt.show()

def display_solution_counts_table(folder_solution_counts):
    """Displays a table showing the count of solutions per folder."""
    folder_indices = list(range(1, len(folder_solution_counts) + 1))
    data = {
        "Folder Index": folder_indices,
        "Folder Name": list(folder_solution_counts.keys()),
        "Solution Count": list(folder_solution_counts.values())
    }
    df = pd.DataFrame(data)
    print("\n=== Solution Counts by Folder ===")
    print(df.to_string(index=False))
    return df

def display_experiment_results_table(global_results):
    """Displays a table summarizing the results of each experiment."""
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
    return df

def display_metric_statistics(global_results, folder_solution_counts):
    """Displays a table with mean and standard deviation of each metric for each folder."""
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
    return df

def main():
    main_folder = input("Enter the main folder path containing RM and S folders: ")

    global_results = {}
    folder_solution_counts = {}
    global_execution_times = {}

    for reference_model_folder in os.listdir(main_folder):
        reference_folder_path = os.path.join(main_folder, reference_model_folder)
        if os.path.isdir(reference_folder_path):
            rm_file = None
            for file in os.listdir(reference_folder_path):
                if file.startswith("RM") and file.endswith(".bpmn"):
                    rm_file = file
                    break

            if rm_file:
                reference_path = os.path.join(reference_folder_path, rm_file)
                bpmn_graph = pm4py.read_bpmn(reference_path)
                ref_net, ref_im, ref_fm = bpmn_converter.apply(bpmn_graph, variant=bpmn_converter.Variants.TO_PETRI_NET)

                reference_log = simulator.apply(ref_net, ref_im, variant=simulator.Variants.BASIC_PLAYOUT,
                                                 parameters={"no_traces": 1})

                folder_results = []
                execution_times = []

                for solution_file in os.listdir(reference_folder_path):
                    if solution_file.startswith("S") and solution_file.endswith(".bpmn"):
                        solution_path = os.path.join(reference_folder_path, solution_file)
                        bpmn_graph_s = pm4py.read_bpmn(solution_path)
                        sol_net, sol_im, sol_fm = bpmn_converter.apply(bpmn_graph_s, variant=bpmn_converter.Variants.TO_PETRI_NET)

                        start_time = time.time()
                        recall, precision, generalization, simplicity = evaluate_model(reference_log, sol_net, sol_im, sol_fm)
                        end_time = time.time()

                        execution_time = end_time - start_time
                        execution_times.append(execution_time)

                        result = {
                            "model": solution_file,
                            "recall": recall,
                            "precision": precision,
                            "generalization": generalization,
                            "simplicity": simplicity,
                        }
                        folder_results.append(result)

                if folder_results:
                    print(f"Results for folder {reference_model_folder}:")
                    for res in folder_results:
                        print(res)
                    plot_results_per_metric(folder_results, reference_model_folder)

                global_results[reference_model_folder] = folder_results
                folder_solution_counts[reference_model_folder] = len(folder_results)
                global_execution_times[reference_model_folder] = execution_times

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

    return solution_counts_df, experiment_results_df, metric_statistics_df

if __name__ == "__main__":
    # main()
    solution_counts_table, experiment_results_table, metric_statistics_table = main()
