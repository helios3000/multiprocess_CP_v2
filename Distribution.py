import multiprocessing as mp


class Distributor(mp.Process):
    def __init__(self,
                 processed_spo2_queue, processed_ibp_queue, processed_pump1_queue,
                 processed_pump2_queue, processed_flow_queue, processed_preload_queue,
                 processed_afterload_queue,

                 worker1_spo2_queue, worker1_ibp_queue, worker1_pump1_queue,
                 worker1_pump2_queue, worker1_flow_queue, worker1_preload_queue,
                 worker1_afterload_queue,

                 worker2_spo2_queue, worker2_ibp_queue, worker2_pump1_queue,
                 worker2_pump2_queue, worker2_flow_queue, worker2_preload_queue,
                 worker2_afterload_queue,

                 worker3_spo2_queue, worker3_ibp_queue, worker3_pump1_queue,
                 worker3_pump2_queue, worker3_flow_queue, worker3_preload_queue,
                 worker3_afterload_queue,

                 worker4_spo2_queue, worker4_ibp_queue, worker4_pump1_queue,
                 worker4_pump2_queue, worker4_flow_queue, worker4_preload_queue,
                 worker4_afterload_queue,

                 semaphore2, worker1_semaphore, worker2_semaphore,
                 worker3_semaphore, worker4_semaphore):

        mp.Process.__init__(self)
        self.distribution_idx = 0

        self.processed_spo2_queue = processed_spo2_queue
        self.processed_ibp_queue = processed_ibp_queue
        self.processed_pump1_queue = processed_pump1_queue
        self.processed_pump2_queue = processed_pump2_queue
        self.processed_flow_queue = processed_flow_queue
        self.processed_preload_queue = processed_preload_queue
        self.processed_afterload_queue = processed_afterload_queue

        self.worker1_spo2_queue = worker1_spo2_queue
        self.worker1_ibp_queue = worker1_ibp_queue
        self.worker1_pump1_queue = worker1_pump1_queue
        self.worker1_pump2_queue = worker1_pump2_queue
        self.worker1_flow_queue = worker1_flow_queue
        self.worker1_preload_queue = worker1_preload_queue
        self.worker1_afterload_queue = worker1_afterload_queue

        self.worker2_spo2_queue = worker2_spo2_queue
        self.worker2_ibp_queue = worker2_ibp_queue
        self.worker2_pump1_queue = worker2_pump1_queue
        self.worker2_pump2_queue = worker2_pump2_queue
        self.worker2_flow_queue = worker2_flow_queue
        self.worker2_preload_queue = worker2_preload_queue
        self.worker2_afterload_queue = worker2_afterload_queue

        self.worker3_spo2_queue = worker3_spo2_queue
        self.worker3_ibp_queue = worker3_ibp_queue
        self.worker3_pump1_queue = worker3_pump1_queue
        self.worker3_pump2_queue = worker3_pump2_queue
        self.worker3_flow_queue = worker3_flow_queue
        self.worker3_preload_queue = worker3_preload_queue
        self.worker3_afterload_queue = worker3_afterload_queue

        self.worker4_spo2_queue = worker4_spo2_queue
        self.worker4_ibp_queue = worker4_ibp_queue
        self.worker4_pump1_queue = worker4_pump1_queue
        self.worker4_pump2_queue = worker4_pump2_queue
        self.worker4_flow_queue = worker4_flow_queue
        self.worker4_preload_queue = worker4_preload_queue
        self.worker4_afterload_queue = worker4_afterload_queue

        self.sema2 = semaphore2
        self.worker1_sema = worker1_semaphore
        self.worker2_sema = worker2_semaphore
        self.worker3_sema = worker3_semaphore
        self.worker4_sema = worker4_semaphore

    def run(self):

        print('Distributor Process start')

        try:
            while True:
                self.sema2.acquire()

                spo2_val = self.processed_spo2_queue.get()
                ibp_val = self.processed_ibp_queue.get()
                pump1_val = self.processed_pump1_queue.get()
                pump2_val = self.processed_pump2_queue.get()
                flow_val = self.processed_flow_queue.get()
                preload_val = self.processed_preload_queue.get()
                afterload_val = self.processed_afterload_queue.get()

                # print(f"[Distributor] Preprocessor로부터 데이터 수신. (IBP: {ibp_val:.4f})")

                target_worker_idx = self.distribution_idx % 4  # 수신한 250Hz 데이터를 62.5Hz 데이터로 4분할

                if target_worker_idx == 0:
                    # print(f"  -> AI_Worker-0 으로 분배 중... (IBP: {ibp_val:.4f})")
                    self.worker1_spo2_queue.put(spo2_val)
                    self.worker1_ibp_queue.put(ibp_val)
                    self.worker1_pump1_queue.put(pump1_val)
                    self.worker1_pump2_queue.put(pump2_val)
                    self.worker1_flow_queue.put(flow_val)
                    self.worker1_preload_queue.put(preload_val)
                    self.worker1_afterload_queue.put(afterload_val)
                    self.worker1_sema.release()

                elif target_worker_idx == 1:
                    # print(f"  -> AI_Worker-1 으로 분배 중... (IBP: {ibp_val:.4f})")
                    self.worker2_spo2_queue.put(spo2_val)
                    self.worker2_ibp_queue.put(ibp_val)
                    self.worker2_pump1_queue.put(pump1_val)
                    self.worker2_pump2_queue.put(pump2_val)
                    self.worker2_flow_queue.put(flow_val)
                    self.worker2_preload_queue.put(preload_val)
                    self.worker2_afterload_queue.put(afterload_val)
                    self.worker2_sema.release()

                elif target_worker_idx == 2:
                    # print(f"  -> AI_Worker-2 으로 분배 중... (IBP: {ibp_val:.4f})")
                    self.worker3_spo2_queue.put(spo2_val)
                    self.worker3_ibp_queue.put(ibp_val)
                    self.worker3_pump1_queue.put(pump1_val)
                    self.worker3_pump2_queue.put(pump2_val)
                    self.worker3_flow_queue.put(flow_val)
                    self.worker3_preload_queue.put(preload_val)
                    self.worker3_afterload_queue.put(afterload_val)
                    self.worker3_sema.release()

                elif target_worker_idx == 3:
                    # print(f"  -> AI_Worker-3 으로 분배 중... (IBP: {ibp_val:.4f})")
                    self.worker4_spo2_queue.put(spo2_val)
                    self.worker4_ibp_queue.put(ibp_val)
                    self.worker4_pump1_queue.put(pump1_val)
                    self.worker4_pump2_queue.put(pump2_val)
                    self.worker4_flow_queue.put(flow_val)
                    self.worker4_preload_queue.put(preload_val)
                    self.worker4_afterload_queue.put(afterload_val)
                    self.worker4_sema.release()

                self.distribution_idx += 1

        except KeyboardInterrupt:
            print("Distributor Process interrupted by user.")
        except Exception as error:
            print(f"An error occurred in Distributor Process: {error}")
