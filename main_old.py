import multiprocessing as mp

from Serial import SerialReceiver
from Accumulation import DataPreprocessor
from AI import ApplyAI
from Determination_upsample import CounterPulsation

# from Graph import GraphProcess

spo2_queue = mp.Queue()
ibp_queue = mp.Queue()
pump1_queue = mp.Queue()
pump2_queue = mp.Queue()
flow_queue = mp.Queue()
preload_queue = mp.Queue()

processed_spo2_queue = mp.Queue()
processed_ibp_queue = mp.Queue()
processed_pump1_queue = mp.Queue()
processed_pump2_queue = mp.Queue()
processed_flow_queue = mp.Queue()
processed_preload_queue = mp.Queue()
processed_afterload_queue = mp.Queue()

save_spo2_queue = mp.Queue()
save_outp_h_spo2_queue = mp.Queue()
save_outp_e_spo2_queue = mp.Queue()

save_ibp_queue = mp.Queue()
save_sac1_queue = mp.Queue()
save_sac2_queue = mp.Queue()
save_outp_h_queue = mp.Queue()
save_outp_e_queue = mp.Queue()
save_flow_queue = mp.Queue()
save_preload_queue = mp.Queue()
save_afterload_queue = mp.Queue()

save_upsample_spo2_queue = mp.Queue()
save_upsample_ibp_queue = mp.Queue()
save_upsample_p1_queue = mp.Queue()
save_upsample_p2_queue = mp.Queue()
save_upsample_flow_queue = mp.Queue()
save_upsample_preload_queue = mp.Queue()
save_upsample_afterload_queue = mp.Queue()

spo2_graph_queue = mp.Queue()
outp_h_spo2_graph_queue = mp.Queue()
outp_e_spo2_graph_queue = mp.Queue()

ibp_graph_queue = mp.Queue()
sac1_graph_queue = mp.Queue()
sac2_graph_queue = mp.Queue()
outp_h_graph_queue = mp.Queue()
outp_e_graph_queue = mp.Queue()

draw_ibp_queue = mp.Queue()
draw_sac1_queue = mp.Queue()
draw_sac2_queue = mp.Queue()
draw_outp_h_queue = mp.Queue()
draw_outp_e_queue = mp.Queue()

if __name__ == "__main__":
    semaphore1 = mp.Semaphore(0)
    semaphore2 = mp.Semaphore(0)
    semaphore3 = mp.Semaphore(0)

    # Create and start the processes
    serial_receiver = SerialReceiver(spo2_queue, ibp_queue, pump1_queue, pump2_queue, flow_queue, preload_queue,
                                     semaphore1)
    data_preprocessor = DataPreprocessor(spo2_queue, ibp_queue, pump1_queue, pump2_queue, flow_queue, preload_queue,
                                         processed_spo2_queue, processed_ibp_queue, processed_pump1_queue,
                                         processed_pump2_queue, processed_flow_queue, processed_preload_queue,
                                         processed_afterload_queue, semaphore1, semaphore2)
    apply_ai = ApplyAI(processed_spo2_queue, processed_ibp_queue, processed_pump1_queue, processed_pump2_queue,
                       processed_flow_queue, processed_preload_queue, processed_afterload_queue, save_spo2_queue,
                       save_outp_h_spo2_queue, save_outp_e_spo2_queue, save_ibp_queue, save_sac1_queue, save_sac2_queue,
                       save_outp_h_queue, save_outp_e_queue, save_flow_queue, save_preload_queue, save_afterload_queue,
                       save_upsample_spo2_queue, save_upsample_ibp_queue, save_upsample_p1_queue,
                       save_upsample_p2_queue, save_upsample_flow_queue, save_upsample_preload_queue,
                       save_upsample_afterload_queue, semaphore2, semaphore3)
    counter_pulsation = CounterPulsation(save_spo2_queue, save_outp_h_spo2_queue, save_outp_e_spo2_queue,
                                         save_ibp_queue, save_sac1_queue, save_sac2_queue,
                                         save_outp_h_queue, save_outp_e_queue, save_flow_queue, save_preload_queue,
                                         save_afterload_queue, save_upsample_spo2_queue, save_upsample_ibp_queue,
                                         save_upsample_p1_queue, save_upsample_p2_queue, save_upsample_flow_queue,
                                         save_upsample_preload_queue, save_upsample_afterload_queue, semaphore3)
    # drawing_graph = GraphProcess(ibp_graph_queue, sac1_graph_queue, sac2_graph_queue, outp_h_graph_queue,
    #                              outp_e_graph_queue)

    processes = [
        mp.Process(target=serial_receiver.run, daemon=True),
        mp.Process(target=data_preprocessor.run, daemon=True),
        mp.Process(target=apply_ai.run, daemon=True),
        mp.Process(target=counter_pulsation.run, daemon=True),
        # mp.Process(target=drawing_graph.run, daemon=True)
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()