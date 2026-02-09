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
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

from pm4py.objects.bpmn.importer import importer as bpmn_importer
from pm4py.objects.conversion.bpmn import converter as bpmn_converter
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
from pm4py.objects.log.obj import EventLog, Trace
from pm4py.objects.log.exporter.xes import exporter as xes_exporter


def gerar_log_sintetico_de_multiplos_bpmn(lista_caminhos_bpmn, saida_xes_path, traces_por_modelo=50):
    log_unificado = EventLog()
    trace_counter = 0  # Contador global de traces

    for caminho in lista_caminhos_bpmn:
        print(f"Simulando: {caminho}")
        bpmn_model = bpmn_importer.apply(caminho)
        petri_net, im, fm = bpmn_converter.apply(bpmn_model)

        log_simulado = simulator.apply(petri_net, im, variant=simulator.Variants.BASIC_PLAYOUT, parameters={
            "no_traces": traces_por_modelo
        })

        for trace in log_simulado:
            # Renumerar o trace com o contador global
            trace.attributes["concept:name"] = f"case_{trace_counter}"
            log_unificado.append(trace)
            trace_counter += 1

    xes_exporter.apply(log_unificado, saida_xes_path)
    print(f"Log sintético salvo em: {saida_xes_path}")
    print(f"Total de traces: {trace_counter}")


# if __name__ == "__main__":
#     main_folder = input("Enter the main folder path containing RM and S folders: ")
#
#     global_results = {}
#     folder_solution_counts = {}
#     global_execution_times = {}
#
#     for reference_model_folder in os.listdir(main_folder):
#         reference_folder_path = os.path.join(main_folder, reference_model_folder)
#         if os.path.isdir(reference_folder_path):
#             rm_file = None
#             for file in os.listdir(reference_folder_path):
#                 if file.startswith("RM") and file.endswith(".bpmn"):
#                     rm_file = file
#                     break
#
#             if rm_file:
#                 reference_path = os.path.join(reference_folder_path, rm_file)
#                 bpmn_graph = pm4py.read_bpmn(reference_path)
#                 ref_net, ref_im, ref_fm = bpmn_converter.apply(bpmn_graph, variant=bpmn_converter.Variants.TO_PETRI_NET)
#
#                 reference_log = simulator.apply(ref_net, ref_im, variant=simulator.Variants.BASIC_PLAYOUT,
#                                                  parameters={"no_traces": 1})
#
#                 export_path = os.path.join(reference_folder_path, rm_file)
#                 base_dir = os.path.dirname(reference_path)
#                 base_name = os.path.splitext(os.path.basename(reference_path))[0]
#                 output_xes_path = os.path.join(base_dir, f"{base_name}.xes")
#
#                 # 5. Exportar log
#                 xes_exporter.apply(reference_log, output_xes_path)

# if __name__ == "__main__":
#     main_folder = input("Enter the main folder path containing RM and S folders: ")

#     global_results = {}
#     folder_solution_counts = {}
#     global_execution_times = {}

#     for reference_model_folder in os.listdir(main_folder):
#         reference_folder_path = os.path.join(main_folder, reference_model_folder)
#         if os.path.isdir(reference_folder_path):
#             s_file = None
#             for file in os.listdir(reference_folder_path):
#                 # if file.startswith("S") and file.endswith(".bpmn"):
#                 if file.endswith(".bpmn"):
#                     s_file = file
#                     # break

#                     if s_file:
#                         reference_path = os.path.join(reference_folder_path, s_file)
#                         bpmn_graph = pm4py.read_bpmn(reference_path)
#                         ref_net, ref_im, ref_fm = bpmn_converter.apply(bpmn_graph, variant=bpmn_converter.Variants.TO_PETRI_NET)

#                         reference_log = simulator.apply(ref_net, ref_im, variant=simulator.Variants.BASIC_PLAYOUT,
#                                                          parameters={"no_traces": 1})

#                         export_path = os.path.join(reference_folder_path, s_file)
#                         base_dir = os.path.dirname(reference_path)
#                         base_name = os.path.splitext(os.path.basename(reference_path))[0]
#                         output_xes_path = os.path.join(base_dir, f"{base_name}.xes")

#                         # 5. Exportar log
#                         xes_exporter.apply(reference_log, output_xes_path)


if __name__ == "__main__":
    main_folder = input("Enter the main folder path containing RM and S folders: ")

    global_results = {}
    folder_solution_counts = {}
    global_execution_times = {}

    for reference_model_folder in os.listdir(main_folder):
        reference_folder_path = os.path.join(main_folder, reference_model_folder)
        if os.path.isdir(reference_folder_path):
            rm_file = []
            for file in os.listdir(reference_folder_path):
                if file.endswith(".bpmn"):
                    arquivo = os.path.join(reference_folder_path, file)
                    rm_file.append(arquivo)

            if len(rm_file) > 0:
                output_xes_path = os.path.join(reference_folder_path, f"log_sintetico_multimodelo.xes")
                gerar_log_sintetico_de_multiplos_bpmn(rm_file, output_xes_path, traces_por_modelo=30)


# Logs 2 modelos
# if __name__ == "__main__":
#     main_folder = input("Enter the main folder path containing RM and S folders: ")

#     global_results = {}
#     folder_solution_counts = {}
#     global_execution_times = {}

#     for reference_model_folder in os.listdir(main_folder):
#         reference_folder_path = os.path.join(main_folder, reference_model_folder)
#         if os.path.isdir(reference_folder_path):
#             rm_file = []
#             i = 0
#             for file in os.listdir(reference_folder_path):
#                 if file.endswith(".bpmn") and not file.startswith("RM"):
#                     arquivo = os.path.join(reference_folder_path, file)
#                     rm_file.append(arquivo)
#                     i += 1
#                     if i >= 2:
#                         break

#             if len(rm_file) > 0:
#                 output_xes_path = os.path.join(reference_folder_path, f"log_sintetico_2_modelos.xes")
#                 gerar_log_sintetico_de_multiplos_bpmn(rm_file, output_xes_path, traces_por_modelo=1)

