import multiprocessing as mp
import threading
# from Function import monitor_memory

from Serial import SerialReceiver
from Accumulation import DataPreprocessor
# from AI import ApplyAI
from Distribution import Distributor
from Work import AI_Worker
from Determination import CounterPulsation


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

worker1_in_queues = [mp.Queue() for _ in range(7)]
worker2_in_queues = [mp.Queue() for _ in range(7)]
worker3_in_queues = [mp.Queue() for _ in range(7)]
worker4_in_queues = [mp.Queue() for _ in range(7)]

worker1_out_queues = [mp.Queue() for _ in range(11)]
worker2_out_queues = [mp.Queue() for _ in range(11)]
worker3_out_queues = [mp.Queue() for _ in range(11)]
worker4_out_queues = [mp.Queue() for _ in range(11)]

if __name__ == "__main__":
    semaphore1 = mp.Semaphore(0)  # Serial -> Preprocessor
    semaphore2 = mp.Semaphore(0)  # Preprocessor -> Distributor
    worker_semaphores = [mp.Semaphore(0) for _ in range(4)]  # Distributor -> AI Worker

    serial_receiver = SerialReceiver(spo2_queue, ibp_queue, pump1_queue, pump2_queue, flow_queue, preload_queue,
                                     semaphore1)

    data_preprocessor = DataPreprocessor(spo2_queue, ibp_queue, pump1_queue, pump2_queue, flow_queue, preload_queue,
                                         processed_spo2_queue, processed_ibp_queue, processed_pump1_queue,
                                         processed_pump2_queue, processed_flow_queue, processed_preload_queue,
                                         processed_afterload_queue, semaphore1, semaphore2)

    distributor = Distributor(
        # 입력 큐 (from Preprocessor)
        processed_spo2_queue, processed_ibp_queue, processed_pump1_queue, processed_pump2_queue,
        processed_flow_queue, processed_preload_queue, processed_afterload_queue,
        # 출력 큐 (to Worker 0)
        worker1_in_queues[0], worker1_in_queues[1], worker1_in_queues[2], worker1_in_queues[3],
        worker1_in_queues[4], worker1_in_queues[5], worker1_in_queues[6],
        # 출력 큐 (to Worker 1)
        worker2_in_queues[0], worker2_in_queues[1], worker2_in_queues[2], worker2_in_queues[3],
        worker2_in_queues[4], worker2_in_queues[5], worker2_in_queues[6],
        # 출력 큐 (to Worker 2)
        worker3_in_queues[0], worker3_in_queues[1], worker3_in_queues[2], worker3_in_queues[3],
        worker3_in_queues[4], worker3_in_queues[5], worker3_in_queues[6],
        # 출력 큐 (to Worker 3)
        worker4_in_queues[0], worker4_in_queues[1], worker4_in_queues[2], worker4_in_queues[3],
        worker4_in_queues[4], worker4_in_queues[5], worker4_in_queues[6],
        # 세마포어
        semaphore2, worker_semaphores[0], worker_semaphores[1], worker_semaphores[2], worker_semaphores[3]
    )

    all_worker_in_queues = [worker1_in_queues, worker2_in_queues, worker3_in_queues, worker4_in_queues]
    all_worker_out_queues = [worker1_out_queues, worker2_out_queues, worker3_out_queues, worker4_out_queues]

    ai_workers = []
    for i in range(4):
        worker_process = AI_Worker(
            worker_id=i,  # <-- 작업자 식별 번호 전달
            worker_in_queues=all_worker_in_queues[i],
            worker_out_queues=all_worker_out_queues[i],
            worker_semaphore=worker_semaphores[i]
        )
        ai_workers.append(worker_process)

    counter_pulsation = CounterPulsation(
        all_worker_out_queues
    )

    process_map = {
        "Serial": serial_receiver,
        "Preproc": data_preprocessor,
        "Dist": distributor,
        "AI-0": ai_workers[0],
        "AI-1": ai_workers[1],
        "AI-2": ai_workers[2],
        "AI-3": ai_workers[3],
        "Determ": counter_pulsation
    }

    # monitor_thread = threading.Thread(target=monitor_memory, args=(process_map, 5), daemon=True)
    # monitor_thread.start()

    processes_to_run = [
        serial_receiver,
        data_preprocessor,
        distributor,
        *ai_workers,
        counter_pulsation
    ]

    for p_name, p_obj in process_map.items():
        p_obj.daemon = True
        p_obj.start()

    for p_obj in process_map.values():
        p_obj.join()
