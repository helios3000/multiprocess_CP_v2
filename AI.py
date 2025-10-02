import multiprocessing as mp
import threading
import pickle
import numpy as np
from queue import Queue


def dnn(x, h1_w, h1_b, h2_w, h2_b, o_w, o_b):
    def dnn_relu(arr):
        for i_dnn in range(0, arr.shape[0]):
            if arr[i_dnn] < 0:
                arr[i_dnn] = 0
            else:
                pass
        return arr

    z1 = np.matmul(x, h1_w) + h1_b
    z1 = np.array(z1.reshape(z1.shape[0] * z1.shape[1]))
    z1 = dnn_relu(z1)
    z2 = np.matmul(z1, h2_w) + h2_b
    z2 = np.array(z2.reshape(z2.shape[0] * z2.shape[1]))
    z2 = dnn_relu(z2)
    z = np.matmul(z2, o_w) + o_b
    z = np.array(z.reshape(z.shape[0] * z.shape[1]))
    outp = np.zeros(len(z))
    outp[np.argmax(z)] = 1

    return outp


def moving_average_dnn(x, window_size):
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(x, weights, 'valid')


def normalize(arr: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    denom = (mx - mn) if (mx - mn) != 0 else eps
    return (arr - mn) / denom


class ApplyAI(mp.Process):
    def __init__(self, processed_spo2_queue, processed_ibp_queue, processed_pump1_queue, processed_pump2_queue,
                 processed_flow_queue, processed_preload_queue, processed_afterload_queue,
                 save_spo2_queue, save_outp_h_spo2_queue, save_outp_e_spo2_queue, save_ibp_queue, save_sac1_queue,
                 save_sac2_queue, save_outp_h_queue, save_outp_e_queue, save_flow_queue, save_preload_queue,
                 save_afterload_queue, save_upsample_spo2_queue, save_upsample_ibp_queue, save_upsample_p1_queue,
                 save_upsample_p2_queue, save_upsample_flow_queue, save_upsample_preload_queue,
                 save_upsample_afterload_queue, semaphore2, semaphore3):

        mp.Process.__init__(self)

        self.dnn_loop_n = 0
        self.sample_idx = 0

        self.spo2_diff_queue_data = 0
        self.ibp_diff_queue_data = 0
        self.pump1_queue_data = 0
        self.pump2_queue_data = 0
        self.flow_queue_data = 0
        self.preload_queue_data = 0
        self.afterload_queue_data = 0

        self.apply_spo2_diff_arr = np.array([])
        self.upsample_spo2_diff_arr = np.array([])
        self.apply_ibp_diff_arr = np.array([])
        self.upsample_ibp_diff_arr = np.array([])
        self.apply_pump1_arr = np.array([])
        self.upsample_pump1_arr = np.array([])
        self.apply_pump2_arr = np.array([])
        self.upsample_pump2_arr = np.array([])
        self.apply_flow_arr = np.array([])
        self.upsample_flow_arr = np.array([])
        self.apply_preload_arr = np.array([])
        self.upsample_preload_arr = np.array([])
        self.apply_afterload_arr = np.array([])
        self.upsample_afterload_arr = np.array([])

        self.processed_spo2_queue = processed_spo2_queue
        self.processed_ibp_queue = processed_ibp_queue
        self.processed_pump1_queue = processed_pump1_queue
        self.processed_pump2_queue = processed_pump2_queue
        self.processed_flow_queue = processed_flow_queue
        self.processed_preload_queue = processed_preload_queue
        self.processed_afterload_queue = processed_afterload_queue

        self.save_spo2 = np.array([])
        self.save_diff = np.array([])
        self.save_sac1 = np.array([])
        self.save_sac2 = np.array([])
        self.save_outp_h = np.array([])
        self.save_outp_e = np.array([])
        self.save_flow = np.array([])
        self.save_preload = np.array([])
        self.save_afterload = np.array([])

        self.save_spo2_queue = save_spo2_queue
        self.save_outp_h_spo2_queue = save_outp_h_spo2_queue
        self.save_outp_e_spo2_queue = save_outp_e_spo2_queue

        self.save_ibp_queue = save_ibp_queue
        self.save_sac1_queue = save_sac1_queue
        self.save_sac2_queue = save_sac2_queue
        self.save_outp_h_queue = save_outp_h_queue
        self.save_outp_e_queue = save_outp_e_queue
        self.save_flow_queue = save_flow_queue
        self.save_preload_queue = save_preload_queue
        self.save_afterload_queue = save_afterload_queue

        self.save_upsample_spo2_queue = save_upsample_spo2_queue
        self.save_upsample_ibp_queue = save_upsample_ibp_queue
        self.save_upsample_pump1_queue = save_upsample_p1_queue
        self.save_upsample_pump2_queue = save_upsample_p2_queue
        self.save_upsample_flow_queue = save_upsample_flow_queue
        self.save_upsample_preload_queue = save_upsample_preload_queue
        self.save_upsample_afterload_queue = save_upsample_afterload_queue

        self.sema2 = semaphore2
        self.sema3 = semaphore3

        save_path = r'C:\Users\user\PycharmProjects\Multiprocess_CP_v2\ecmo_ai_model_221007.pickle'

        with open(save_path, 'rb') as f:
            training_db = pickle.load(f)

        self.w1 = training_db['heart']['w1']
        self.w2 = training_db['heart']['w2']
        self.w3 = training_db['heart']['w3']
        self.w4 = training_db['ecmo']['w4']
        self.w5 = training_db['ecmo']['w5']
        self.w6 = training_db['ecmo']['w6']

        self.b1 = training_db['heart']['b1']
        self.b2 = training_db['heart']['b2']
        self.b3 = training_db['heart']['b3']
        self.b4 = training_db['ecmo']['b4']
        self.b5 = training_db['ecmo']['b5']
        self.b6 = training_db['ecmo']['b6']

    def run(self):

        print("DNN apply start")

        self.apply_inp_spo2 = Queue()
        self.apply_spo2_norm = Queue()
        self.apply_inp_ibp = Queue()
        self.apply_ibp_norm = Queue()
        self.apply_sac1_seg = Queue()
        self.apply_sac2_seg = Queue()
        self.apply_flow_seg = Queue()
        self.apply_preload_seg = Queue()
        self.apply_afterload_seg = Queue()

        self.upsample_inp_spo2 = Queue()
        self.upsample_spo2_norm = Queue()
        self.upsample_inp_ibp = Queue()
        self.upsample_ibp_norm = Queue()
        self.upsample_sac1_seg = Queue()
        self.upsample_sac2_seg = Queue()
        self.upsample_flow_seg = Queue()
        self.upsample_preload_seg = Queue()
        self.upsample_afterload_seg = Queue()

        self.apply_data_queue = Queue()
        self.upsample_data_queue = Queue()

        def ai_data_modify():
            try:
                while True:

                    self.sema2.acquire()

                    if not self.processed_ibp_queue.empty():
                        self.spo2_diff_queue_data = self.processed_spo2_queue.get()
                        self.ibp_diff_queue_data = self.processed_ibp_queue.get()
                        self.pump1_queue_data = self.processed_pump1_queue.get()
                        self.pump2_queue_data = self.processed_pump2_queue.get()
                        self.flow_queue_data = self.processed_flow_queue.get()
                        self.preload_queue_data = self.processed_preload_queue.get()
                        self.afterload_queue_data = self.processed_afterload_queue.get()

                    if self.sample_idx % 4 in (0, 1, 2):
                        # part1: 업샘플링 복원용
                        self.upsample_spo2_diff_arr = np.append(self.upsample_spo2_diff_arr, self.spo2_diff_queue_data)
                        self.upsample_ibp_diff_arr = np.append(self.upsample_ibp_diff_arr, self.ibp_diff_queue_data)
                        self.upsample_pump1_arr = np.append(self.upsample_pump1_arr, self.pump1_queue_data)
                        self.upsample_pump2_arr = np.append(self.upsample_pump2_arr, self.pump2_queue_data)
                        self.upsample_flow_arr = np.append(self.upsample_flow_arr, self.flow_queue_data)
                        self.upsample_preload_arr = np.append(self.upsample_preload_arr, self.preload_queue_data)
                        self.upsample_afterload_arr = np.append(self.upsample_afterload_arr, self.afterload_queue_data)

                        if len(self.upsample_ibp_diff_arr) > 127:
                            self.upsample_spo2_diff_arr = np.array(self.upsample_spo2_diff_arr[1::])
                            self.upsample_ibp_diff_arr = np.array(self.upsample_ibp_diff_arr[1::])
                            self.upsample_pump1_arr = np.array(self.upsample_pump1_arr[1::])
                            self.upsample_pump2_arr = np.array(self.upsample_pump2_arr[1::])
                            self.upsample_flow_arr = np.array(self.upsample_flow_arr[1::])
                            self.upsample_preload_arr = np.array(self.upsample_preload_arr[1::])
                            self.upsample_afterload_arr = np.array(self.upsample_afterload_arr[1::])

                    elif self.sample_idx % 4 == 3:
                        # part2: DNN 입력용
                        self.apply_spo2_diff_arr = np.append(self.apply_spo2_diff_arr, self.spo2_diff_queue_data)
                        self.apply_ibp_diff_arr = np.append(self.apply_ibp_diff_arr, self.ibp_diff_queue_data)
                        self.apply_pump1_arr = np.append(self.apply_pump1_arr, self.pump1_queue_data)
                        self.apply_pump2_arr = np.append(self.apply_pump2_arr, self.pump2_queue_data)
                        self.apply_flow_arr = np.append(self.apply_flow_arr, self.flow_queue_data)
                        self.apply_preload_arr = np.append(self.apply_preload_arr, self.preload_queue_data)
                        self.apply_afterload_arr = np.append(self.apply_afterload_arr, self.afterload_queue_data)

                        if len(self.apply_ibp_diff_arr) > 127:
                            self.apply_spo2_diff_arr = np.array(self.apply_spo2_diff_arr[1::])
                            self.apply_ibp_diff_arr = np.array(self.apply_ibp_diff_arr[1::])
                            self.apply_pump1_arr = np.array(self.apply_pump1_arr[1::])
                            self.apply_pump2_arr = np.array(self.apply_pump2_arr[1::])
                            self.apply_flow_arr = np.array(self.apply_flow_arr[1::])
                            self.apply_preload_arr = np.array(self.apply_preload_arr[1::])
                            self.apply_afterload_arr = np.array(self.apply_afterload_arr[1::])

                    if len(self.apply_ibp_diff_arr) > 126:
                        for version in ('upsample', 'apply'):
                            # 1) segment slicing
                            spo2_seg = np.array(getattr(self, f"{version}_spo2_diff_arr")[:127], dtype='float32')
                            ibp_seg = np.array(getattr(self, f"{version}_ibp_diff_arr")[:127], dtype='float32')
                            sac1_seg = np.array(getattr(self, f"{version}_pump1_arr")[37:127], dtype='float32')
                            sac2_seg = np.array(getattr(self, f"{version}_pump2_arr")[37:127], dtype='float32')
                            flow_seg = np.array(getattr(self, f"{version}_flow_arr")[:125], dtype='float32')
                            preload_seg = np.array(getattr(self, f"{version}_preload_arr")[:125], dtype='float32')
                            afterload_seg = np.array(getattr(self, f"{version}_afterload_arr")[:125], dtype='float32')

                            # 2) normalize + MA filter (for diff arrays)
                            spo2_norm = moving_average_dnn(normalize(spo2_seg), 3)
                            ibp_norm = moving_average_dnn(normalize(ibp_seg), 3)

                            # 3) DNN input 구성
                            inp_spo2 = np.hstack((spo2_norm, sac1_seg, sac2_seg))
                            inp_ibp = np.hstack((ibp_norm, sac1_seg, sac2_seg))

                            # 4) 변수별로 바로 각 큐에 put()
                            getattr(self, f"{version}_inp_spo2").put(inp_spo2)
                            getattr(self, f"{version}_spo2_norm").put(spo2_norm)
                            getattr(self, f"{version}_inp_ibp").put(inp_ibp)
                            getattr(self, f"{version}_ibp_norm").put(ibp_norm)
                            getattr(self, f"{version}_sac1_seg").put(sac1_seg)
                            getattr(self, f"{version}_sac2_seg").put(sac2_seg)
                            getattr(self, f"{version}_flow_seg").put(flow_seg)
                            getattr(self, f"{version}_preload_seg").put(preload_seg)
                            getattr(self, f"{version}_afterload_seg").put(afterload_seg)

                    self.sample_idx += 1

            except KeyboardInterrupt:
                print("Apply ai_data_modify interrupted by user.")
            except Exception as error:
                print("An error occurred in ai_data_modify Process:", str(error))

        def ai_apply():
            try:
                while True:

                    inp_spo2 = self.apply_inp_spo2.get()
                    inp_ibp = self.apply_inp_ibp.get()

                    save_diff_spo2_val = self.apply_spo2_norm.get()
                    save_diff_val = self.apply_ibp_norm.get()
                    save_sac1_val = self.apply_sac1_seg.get()
                    save_sac2_val = self.apply_sac2_seg.get()
                    save_flow_val = self.apply_flow_seg.get()
                    save_preload_val = self.apply_preload_seg.get()
                    save_afterload_val = self.apply_afterload_seg.get()

                    save_upsample_spo2_val = self.upsample_spo2_norm.get()
                    save_upsample_ibp_val = self.upsample_ibp_norm.get()
                    save_upsample_pump1_val = self.upsample_sac1_seg.get()
                    save_upsample_pump2_val = self.upsample_sac2_seg.get()
                    save_upsample_flow_val = self.upsample_flow_seg.get()
                    save_upsample_preload_val = self.upsample_preload_seg.get()
                    save_upsample_afterload_val = self.upsample_afterload_seg.get()

                    # spo2 DNN 적용
                    outp_h_spo2 = dnn(inp_spo2, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3)
                    outp_e_spo2 = dnn(inp_spo2, self.w4, self.b4, self.w5, self.b5, self.w6, self.b6)

                    outp_h_spo2 = np.array(outp_h_spo2[0:-1])
                    outp_e_spo2 = np.array(outp_e_spo2[0:-1])

                    # ibp DNN 적용
                    outp_h = dnn(inp_ibp, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3)
                    outp_e = dnn(inp_ibp, self.w4, self.b4, self.w5, self.b5, self.w6, self.b6)

                    outp_h = np.array(outp_h[0:-1])
                    outp_e = np.array(outp_e[0:-1])

                    # Accumulation DNN outcome
                    # 최초 데이터 누적 시 65개 old data 저장, 이후 누적 판정이 끝난 60번부터 data 저장
                    if self.dnn_loop_n == 0:

                        self.save_outp_h_spo2 = np.hstack((np.zeros(65), outp_h_spo2, np.zeros(30)))
                        self.save_outp_e_spo2 = np.hstack((np.zeros(65), outp_e_spo2, np.zeros(30)))

                        self.save_outp_h = np.hstack((np.zeros(65), outp_h, np.zeros(30)))
                        self.save_outp_e = np.hstack((np.zeros(65), outp_e, np.zeros(30)))

                        self.save_spo2 = np.array(save_diff_spo2_val)
                        self.save_diff = np.array(save_diff_val)
                        self.save_sac1 = np.hstack((np.zeros(35), save_sac1_val))
                        self.save_sac2 = np.hstack((np.zeros(35), save_sac2_val))
                        self.save_flow = np.array(save_flow_val)
                        self.save_preload = np.array(save_preload_val)
                        self.save_afterload = np.array(save_afterload_val)

                        self.save_upsample_spo2 = np.array(save_upsample_spo2_val)
                        self.save_upsample_ibp = np.array(save_upsample_ibp_val)
                        self.save_upsample_pump1 = np.array(save_upsample_pump1_val)
                        self.save_upsample_pump2 = np.array(save_upsample_pump2_val)
                        self.save_upsample_flow = np.array(save_upsample_flow_val)
                        self.save_upsample_prelaod = np.array(save_upsample_preload_val)
                        self.save_upsample_afterload = np.array(save_upsample_afterload_val)

                        for i in range(0, 65):
                            self.save_outp_h_spo2_queue.put(self.save_outp_h_spo2[i])
                            self.save_outp_e_spo2_queue.put(self.save_outp_e_spo2[i])

                            self.save_outp_h_queue.put(self.save_outp_h[i])
                            self.save_outp_e_queue.put(self.save_outp_e[i])

                            self.save_spo2_queue.put(save_diff_spo2_val[i])
                            self.save_ibp_queue.put(save_diff_val[i])
                            self.save_sac1_queue.put(self.save_sac1[i])
                            self.save_sac2_queue.put(self.save_sac2[i])
                            self.save_flow_queue.put(save_flow_val[i])
                            self.save_preload_queue.put(save_preload_val[i])
                            self.save_afterload_queue.put(save_afterload_val[i])

                            self.save_upsample_spo2_queue.put(save_upsample_spo2_val[i])
                            self.save_upsample_ibp_queue.put(save_upsample_ibp_val[i])
                            self.save_upsample_pump1_queue.put(save_upsample_pump1_val[i])
                            self.save_upsample_pump2_queue.put(save_upsample_pump2_val[i])
                            self.save_upsample_flow_queue.put(save_upsample_flow_val[i])
                            self.save_upsample_preload_queue.put(save_upsample_preload_val[i])
                            self.save_upsample_afterload_queue.put(save_upsample_afterload_val[i])

                    else:
                        self.save_outp_h_spo2 = np.append(self.save_outp_h_spo2, 0)
                        self.save_outp_h_spo2[-60:-30] = self.save_outp_h_spo2[-60:-30] + outp_h_spo2
                        if len(self.save_outp_h_spo2) > 300:
                            self.save_outp_h_spo2 = self.save_outp_h_spo2[-300:]

                        self.save_outp_e_spo2 = np.append(self.save_outp_e_spo2, 0)
                        self.save_outp_e_spo2[-60:-30] = self.save_outp_e_spo2[-60:-30] + outp_e_spo2
                        if len(self.save_outp_e_spo2) > 300:
                            self.save_outp_e_spo2 = self.save_outp_e_spo2[-300:]

                        self.save_outp_h = np.append(self.save_outp_h, 0)
                        self.save_outp_h[-60:-30] = self.save_outp_h[-60:-30] + outp_h
                        if len(self.save_outp_h) > 300:
                            self.save_outp_h = self.save_outp_h[-300:]

                        self.save_outp_e = np.append(self.save_outp_e, 0)
                        self.save_outp_e[-60:-30] = self.save_outp_e[-60:-30] + outp_e
                        if len(self.save_outp_e) > 300:
                            self.save_outp_e = self.save_outp_e[-300:]

                        self.save_spo2 = np.append(self.save_spo2, save_diff_spo2_val[-1])
                        if len(self.save_spo2) > 300:
                            self.save_spo2 = self.save_spo2[-300:]
                        self.save_diff = np.append(self.save_diff, save_diff_val[-1])
                        if len(self.save_diff) > 300:
                            self.save_diff = self.save_diff[-300:]
                        self.save_sac1 = np.append(self.save_sac1, save_sac1_val[-1])
                        if len(self.save_sac1) > 300:
                            self.save_sac1 = self.save_sac1[-300:]
                        self.save_sac2 = np.append(self.save_sac2, save_sac2_val[-1])
                        if len(self.save_sac2) > 300:
                            self.save_sac2 = self.save_sac2[-300:]
                        self.save_flow = np.append(self.save_flow, save_flow_val[-1])
                        if len(self.save_flow) > 300:
                            self.save_flow = self.save_flow[-300:]
                        self.save_preload = np.append(self.save_preload, save_preload_val[-1])
                        if len(self.save_preload) > 300:
                            self.save_preload = self.save_preload[-300:]
                        self.save_afterload = np.append(self.save_afterload, save_afterload_val[-1])
                        if len(self.save_afterload) > 300:
                            self.save_afterload = self.save_afterload[-300:]

                        self.save_upsample_spo2 = np.append(self.save_upsample_spo2, save_upsample_spo2_val[-1])
                        if len(self.save_upsample_spo2) > 300:
                            self.save_upsample_spo2 = self.save_upsample_spo2[-300:]
                        self.save_upsample_ibp = np.append(self.save_upsample_ibp, save_upsample_ibp_val[-1])
                        if len(self.save_upsample_ibp) > 300:
                            self.save_upsample_ibp = self.save_upsample_ibp[-300:]
                        self.save_upsample_pump1 = np.append(self.save_upsample_pump1, save_upsample_pump1_val[-1])
                        if len(self.save_upsample_pump1) > 300:
                            self.save_upsample_pump1 = self.save_upsample_pump1[-300:]
                        self.save_upsample_pump2 = np.append(self.save_upsample_pump2, save_upsample_pump2_val[-1])
                        if len(self.save_upsample_pump2) > 300:
                            self.save_upsample_pump2 = self.save_upsample_pump2[-300:]
                        self.save_upsample_flow = np.append(self.save_upsample_flow, save_upsample_flow_val[-1])
                        if len(self.save_upsample_flow) > 300:
                            self.save_upsample_flow = self.save_upsample_flow[-300:]
                        self.save_upsample_prelaod = np.append(self.save_upsample_prelaod, save_upsample_flow_val[-1])
                        if len(self.save_upsample_prelaod) > 300:
                            self.save_upsample_prelaod = self.save_upsample_prelaod[-300:]
                        self.save_upsample_afterload = np.append(self.save_upsample_afterload,
                                                                 save_upsample_afterload_val[-1])
                        if len(self.save_upsample_afterload) > 300:
                            self.save_upsample_afterload = self.save_upsample_afterload[-300:]

                        # 데이터 전송
                        self.save_outp_h_spo2_queue.put(self.save_outp_h_spo2[-65])
                        self.save_outp_e_spo2_queue.put(self.save_outp_e_spo2[-65])
                        self.save_outp_h_queue.put(self.save_outp_h[-65])
                        self.save_outp_e_queue.put(self.save_outp_e[-65])

                        self.save_spo2_queue.put(save_diff_spo2_val[-65])

                        self.save_ibp_queue.put(save_diff_val[-65])
                        self.save_sac1_queue.put(save_sac1_val[-65])
                        self.save_sac2_queue.put(save_sac2_val[-65])
                        self.save_flow_queue.put(save_flow_val[-65])
                        self.save_preload_queue.put(save_preload_val[-65])
                        self.save_afterload_queue.put(save_afterload_val[-65])

                        for i in range(3):
                            self.save_upsample_spo2_queue.put(save_upsample_spo2_val[-65 + i])
                            self.save_upsample_ibp_queue.put(save_upsample_ibp_val[-65 + i])
                            self.save_upsample_pump1_queue.put(save_upsample_pump1_val[-65 + i])
                            self.save_upsample_pump2_queue.put(save_upsample_pump2_val[-65 + i])
                            self.save_upsample_flow_queue.put(save_upsample_flow_val[-65 + i])
                            self.save_upsample_preload_queue.put(save_upsample_preload_val[-65 + i])
                            self.save_upsample_afterload_queue.put(save_upsample_afterload_val[-65 + i])

                    # self.sema3.release()
                    self.dnn_loop_n += 1

            except KeyboardInterrupt:
                print("ai_apply Process interrupted by user.")
            except Exception as error:
                print("An error occurred in ai_apply Process:", str(error))

        thread1 = threading.Thread(target=ai_data_modify)
        thread2 = threading.Thread(target=ai_apply)

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()
