import multiprocessing as mp
import numpy as np
import pickle
from Function import dnn, moving_average_dnn, normalize


class AI_Worker(mp.Process):
    def __init__(self, worker_id, worker_in_queues, worker_out_queues, worker_semaphore):

        mp.Process.__init__(self)
        self.worker_id = worker_id

        self.in_spo2_queue = worker_in_queues[0]
        self.in_ibp_queue = worker_in_queues[1]
        self.in_pump1_queue = worker_in_queues[2]
        self.in_pump2_queue = worker_in_queues[3]
        self.in_flow_queue = worker_in_queues[4]
        self.in_preload_queue = worker_in_queues[5]
        self.in_afterload_queue = worker_in_queues[6]

        self.out_spo2_queue = worker_out_queues[0]
        self.out_ibp_queue = worker_out_queues[1]
        self.out_sac1_queue = worker_out_queues[2]
        self.out_sac2_queue = worker_out_queues[3]
        self.out_flow_queue = worker_out_queues[4]
        self.out_preload_queue = worker_out_queues[5]
        self.out_afterload_queue = worker_out_queues[6]
        self.out_h_spo2_queue = worker_out_queues[7]
        self.out_e_spo2_queue = worker_out_queues[8]
        self.out_h_queue = worker_out_queues[9]
        self.out_e_queue = worker_out_queues[10]

        self.worker_sema = worker_semaphore

        self.dnn_loop_n = 0
        self.spo2_diff_window = np.array([])
        self.ibp_diff_window = np.array([])
        self.pump1_window = np.array([])
        self.pump2_window = np.array([])
        self.flow_window = np.array([])
        self.preload_window = np.array([])
        self.afterload_window = np.array([])

        self.save_outp_h_spo2 = np.array([])
        self.save_outp_e_spo2 = np.array([])
        self.save_outp_h = np.array([])
        self.save_outp_e = np.array([])

        # f-NN 학습 모델 로드
        save_path = r'C:\Users\user\PycharmProjects\Multiprocess_CP_v2\fNN_model.pickle'

        with open(save_path, 'rb') as f:
            training_db = pickle.load(f)
        self.w1, self.b1 = training_db['heart']['w1'], training_db['heart']['b1']
        self.w2, self.b2 = training_db['heart']['w2'], training_db['heart']['b2']
        self.w3, self.b3 = training_db['heart']['w3'], training_db['heart']['b3']
        self.w4, self.b4 = training_db['ecmo']['w4'], training_db['ecmo']['b4']
        self.w5, self.b5 = training_db['ecmo']['w5'], training_db['ecmo']['b5']
        self.w6, self.b6 = training_db['ecmo']['w6'], training_db['ecmo']['b6']

    def run(self):
        print(f'AI_Worker-{self.worker_id} Process start')

        try:
            while True:
                self.worker_sema.acquire()

                spo2_val = self.in_spo2_queue.get()
                ibp_val = self.in_ibp_queue.get()
                pump1_val = self.in_pump1_queue.get()
                pump2_val = self.in_pump2_queue.get()
                flow_val = self.in_flow_queue.get()
                preload_val = self.in_preload_queue.get()
                afterload_val = self.in_afterload_queue.get()

                self.ibp_diff_window = np.append(self.ibp_diff_window, ibp_val)
                if len(self.ibp_diff_window) > 127:  # 입력층의 ibp 데이터는 125개이지만 MA filter 적용을 위해 127로 설정
                    self.ibp_diff_window = self.ibp_diff_window[1:]  # old data 삭제
                self.spo2_diff_window = np.append(self.spo2_diff_window, spo2_val)
                if len(self.spo2_diff_window) > 127: self.spo2_diff_window = self.spo2_diff_window[1:]
                self.pump1_window = np.append(self.pump1_window, pump1_val)
                if len(self.pump1_window) > 127: self.pump1_window = self.pump1_window[1:]
                self.pump2_window = np.append(self.pump2_window, pump2_val)
                if len(self.pump2_window) > 127: self.pump2_window = self.pump2_window[1:]
                self.flow_window = np.append(self.flow_window, flow_val)
                if len(self.flow_window) > 127: self.flow_window = self.flow_window[1:]
                self.preload_window = np.append(self.preload_window, preload_val)
                if len(self.preload_window) > 127: self.preload_window = self.preload_window[1:]
                self.afterload_window = np.append(self.afterload_window, afterload_val)
                if len(self.afterload_window) > 127: self.afterload_window = self.afterload_window[1:]

                if len(self.ibp_diff_window) == 127:

                    spo2_seg = np.array(self.spo2_diff_window, dtype='float32')  # 입력층 spo2 데이터
                    ibp_seg = np.array(self.ibp_diff_window, dtype='float32')   # 입력층 ibp 데이터
                    sac1_seg = np.array(self.pump1_window[37:127], dtype='float32')  # 입력층 ECMO pump 1 데이터
                    sac2_seg = np.array(self.pump2_window[37:127], dtype='float32')  # 입력층 ECMO pump 2 데이터

                    spo2_norm = moving_average_dnn(normalize(spo2_seg), 3)  # MA filter 적용
                    ibp_norm = moving_average_dnn(normalize(ibp_seg), 3)  # MA filter 적용

                    inp_spo2 = np.hstack((spo2_norm, sac1_seg, sac2_seg))  # 입력층 (spo2 125 + pump1 90 + pump2 90)
                    inp_ibp = np.hstack((ibp_norm, sac1_seg, sac2_seg))  # 입력층 (ibp 125 + pump1 90 + pump2 90)

                    outp_h_spo2 = dnn(inp_spo2, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3)[0:-1]  # spo2 데이터 신경망 적용, 심장 펄스 판정
                    outp_e_spo2 = dnn(inp_spo2, self.w4, self.b4, self.w5, self.b5, self.w6, self.b6)[0:-1]  # spo2 데이터 신경망 적용, ECMO 펄스 판정
                    outp_h = dnn(inp_ibp, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3)[0:-1]  # ibp 데이터 신경망 적용, 심장 펄스 판정
                    outp_e = dnn(inp_ibp, self.w4, self.b4, self.w5, self.b5, self.w6, self.b6)[0:-1]  # ibp 데이터 신경망 적용, ECMO 펄스 판정

                    if self.dnn_loop_n == 0:
                        self.save_outp_h_spo2 = np.hstack((np.zeros(65), outp_h_spo2, np.zeros(30)))  # 첫 spo2 데이터 누적, 125개 데이터 중 중앙 30개 심장 펄스 결과 및 나머지 0
                        self.save_outp_e_spo2 = np.hstack((np.zeros(65), outp_e_spo2, np.zeros(30)))  # 첫 spo2 데이터 누적, 125개 데이터 중 중앙 30개 ECMO 펄스 결과 및 나머지 0
                        self.save_outp_h = np.hstack((np.zeros(65), outp_h, np.zeros(30)))  # 첫 ibp 데이터 누적, 125개 데이터 중 중앙 30개 심장 펄스 결과 및 나머지 0
                        self.save_outp_e = np.hstack((np.zeros(65), outp_e, np.zeros(30)))  # 첫 ibp 데이터 누적, 125개 데이터 중 중앙 30개 ECMO 펄스 결과 및 나머지 0
                    else:
                        self.save_outp_h_spo2 = np.append(self.save_outp_h_spo2, 0)  # 이후 spo2 데이터 중 심장 펄스 결과 누적
                        self.save_outp_h_spo2[-60:-30] = self.save_outp_h_spo2[-60:-30] + outp_h_spo2  # 심장 펄스 결과 + 길이를 맞추기 위한 0
                        if len(self.save_outp_h_spo2) > 300: self.save_outp_h_spo2 = self.save_outp_h_spo2[-300:]  # 메모리 관리를 위한 길이 조절
                        self.save_outp_e_spo2 = np.append(self.save_outp_e_spo2, 0)  # 이후 spo2 데이터 중 ECMO 펄스 결과 누적
                        self.save_outp_e_spo2[-60:-30] = self.save_outp_e_spo2[-60:-30] + outp_e_spo2  # ECMO 펄스 결과 + 길이를 맞추기 위한 0
                        if len(self.save_outp_e_spo2) > 300: self.save_outp_e_spo2 = self.save_outp_e_spo2[-300:]  # 메모리 관리를 위한 길이 조절
                        self.save_outp_h = np.append(self.save_outp_h, 0)  # 이후 ibp 데이터 중 심장 펄스 결과 누적
                        self.save_outp_h[-60:-30] = self.save_outp_h[-60:-30] + outp_h  # 심장 펄스 결과 + 길이를 맞추기 위한 0
                        if len(self.save_outp_h) > 300: self.save_outp_h = self.save_outp_h[-300:]  # 메모리 관리를 위한 길이 조절
                        self.save_outp_e = np.append(self.save_outp_e, 0)  # 이후 ibp 데이터 중 ECMO 펄스 결과 누적
                        self.save_outp_e[-60:-30] = self.save_outp_e[-60:-30] + outp_e  # ECMO 펄스 결과 + 길이를 맞추기 위한 0
                        if len(self.save_outp_e) > 300: self.save_outp_e = self.save_outp_e[-300:]  # 메모리 관리를 위한 길이 조절

                    # 결과 데이터 Queue에 저장
                    self.out_spo2_queue.put(spo2_norm[-65])
                    self.out_ibp_queue.put(ibp_norm[-65])
                    self.out_sac1_queue.put(sac1_seg[-65])
                    self.out_sac2_queue.put(sac2_seg[-65])
                    self.out_flow_queue.put(self.flow_window[-65])
                    self.out_preload_queue.put(self.preload_window[-65])
                    self.out_afterload_queue.put(self.afterload_window[-65])

                    self.out_h_spo2_queue.put(self.save_outp_h_spo2[-65])
                    self.out_e_spo2_queue.put(self.save_outp_e_spo2[-65])
                    self.out_h_queue.put(self.save_outp_h[-65])
                    self.out_e_queue.put(self.save_outp_e[-65])

                    self.dnn_loop_n += 1

        except KeyboardInterrupt:
            print(f"AI_Worker-{self.worker_id} Process interrupted by user.")
        except Exception as error:
            print(f"An error occurred in AI_Worker-{self.worker_id} Process: {error}")