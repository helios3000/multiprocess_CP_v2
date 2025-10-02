import multiprocessing as mp
import numpy as np
from Function import moving_average, derivation


class DataPreprocessor(mp.Process):
    def __init__(self, spo2_queue, ibp_queue, pump1_queue, pump2_queue, flow_queue, preload_queue, processed_spo2_queue,
                 processed_ibp_queue, processed_pump1_queue, processed_pump2_queue, processed_flow_queue,
                 processed_preload_queue, processed_afterload_queue, semaphore1, semaphore2):
        mp.Process.__init__(self)

        self.downsample_flag = False

        self.spo2_queue = spo2_queue
        self.ibp_queue = ibp_queue
        self.pump1_queue = pump1_queue
        self.pump2_queue = pump2_queue
        self.flow_queue = flow_queue
        self.preload_queue = preload_queue

        self.spo2_val = 0
        self.ibp_val = 0
        self.pump1_val = 0
        self.pump2_val = 0
        self.flow_val = 0
        self.preload_val = 0

        self.spo2_wave_arr = ([])
        self.spo2_filtered_arr = ([])
        self.spo2_diff_tmp_arr =([])
        self.spo2_diff_arr = ([])
        self.ibp_wave_arr = ([])
        self.ibp_filtered_arr = ([])
        self.ibp_diff_tmp_arr = ([])
        self.ibp_diff_arr = ([])
        self.pump1_arr = ([])
        self.pump2_arr = ([])
        self.flow_arr = ([])
        self.preload_arr = ([])

        self.processed_spo2_queue = processed_spo2_queue
        self.processed_ibp_queue = processed_ibp_queue
        self.processed_pump1_queue = processed_pump1_queue
        self.processed_pump2_queue = processed_pump2_queue
        self.processed_flow_queue = processed_flow_queue
        self.processed_preload_queue = processed_preload_queue
        self.processed_afterload_queue = processed_afterload_queue

        self.down_sampling_loop_n = 0

        self.sema1 = semaphore1
        self.sema2 = semaphore2

    def run(self):

        print('Data Preprocessor start')

        try:
            while True:

                self.sema1.acquire()
                # time.sleep(0.001)

                self.spo2_val = self.spo2_queue.get()
                self.ibp_val = self.ibp_queue.get()
                self.pump1_val = self.pump1_queue.get()
                self.pump2_val = self.pump2_queue.get()
                self.flow_val = self.flow_queue.get()
                self.preload_val = self.preload_queue.get()

                if self.pump1_val == '0':
                    self.pump1_val = 1
                else:
                    self.pump1_val = 0

                if self.pump2_val == '0':
                    self.pump2_val = 1
                else:
                    self.pump2_val = 0

                if not self.downsample_flag or self.down_sampling_loop_n % 2 == 0:  # Downsampling ON/OFF

                    self.spo2_wave_arr = np.append(self.spo2_wave_arr, self.spo2_val)   # spo2 array
                    if len(self.spo2_wave_arr) > 10:
                        self.spo2_wave_arr = self.spo2_wave_arr[1:]  # old data 삭제, 메모리 관리
                    # print("spo2_wave:", *self.spo2_wave_arr)

                    self.ibp_wave_arr = np.append(self.ibp_wave_arr, self.ibp_val)  # ibp array
                    if len(self.ibp_wave_arr) > 10:
                        self.ibp_wave_arr = self.ibp_wave_arr[1:]  # old data 삭제, 메모리 관리
                    # print("ibp_wave:", *self.ibp_wave_arr)

                    if len(self.spo2_wave_arr) >= 5:
                        self.spo2_filtered_arr = np.append(self.spo2_filtered_arr, moving_average(self.spo2_wave_arr))  # MA filter 적용
                        if len(self.spo2_filtered_arr) > 10:
                            self.spo2_filtered_arr = self.spo2_filtered_arr[1:]  # old data 삭제, 메모리 관리
                        # print("spo2_filtered:", *self.spo2_filtered_arr)

                    if len(self.ibp_wave_arr) >= 5:
                        self.ibp_filtered_arr = np.append(self.ibp_filtered_arr, moving_average(self.ibp_wave_arr))  # MA filter 적용
                        if len(self.ibp_filtered_arr) > 10:
                            self.ibp_filtered_arr = self.ibp_filtered_arr[1:]  # old data 삭제, 메모리 관리
                        # print("ibp_filtered:", *self.ibp_filtered_arr)

                        if len(self.spo2_filtered_arr) >= 2:
                            self.spo2_diff_tmp_arr = np.append(self.spo2_diff_tmp_arr,
                                                               derivation(self.spo2_filtered_arr[-1],
                                                                          self.spo2_filtered_arr[-2]))  # spo2 데이터 미분
                            if len(self.spo2_diff_tmp_arr) > 10:
                                self.spo2_diff_tmp_arr = self.spo2_diff_tmp_arr[1:]  # old data 삭제, 메모리 관리
                            # print("spo2_diff: ", *self.spo2_diff_tmp_arr)

                        if len(self.ibp_filtered_arr) >= 2:
                            self.ibp_diff_tmp_arr = np.append(self.ibp_diff_tmp_arr,
                                                              derivation(self.ibp_filtered_arr[-1],
                                                                         self.ibp_filtered_arr[-2]))  # ibp 데이터 미분
                            if len(self.ibp_diff_tmp_arr) > 10:
                                self.ibp_diff_tmp_arr = self.ibp_diff_tmp_arr[1:]  # old data 삭제, 메모리 관리
                            # print("ibp_diff: ", *self.ibp_diff_tmp_arr)
                            # print('')

                            # data saving to queue
                            self.processed_spo2_queue.put(self.spo2_diff_tmp_arr[-1])
                            self.processed_ibp_queue.put(self.ibp_diff_tmp_arr[-1])
                            self.processed_pump1_queue.put(self.pump1_val)
                            self.processed_pump2_queue.put(self.pump2_val)
                            self.processed_flow_queue.put(self.flow_val)
                            self.processed_preload_queue.put(self.preload_val)
                            self.processed_afterload_queue.put(self.ibp_val)

                            self.sema2.release()

                self.down_sampling_loop_n += 1

        except KeyboardInterrupt:
            print("Data Processing Process interrupted by user.")
        except Exception as error:
            print("An error occurred in Data Processing Process:", str(error))
