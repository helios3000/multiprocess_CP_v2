import multiprocessing as mp
import threading
import serial
import numpy as np
from queue import Queue
import time


# alternating_update_array (apply + upsample 교차 저장)
def alternating_update_array(target_array, new_data_list, max_len=500):
    # np.concatenate는 여러 배열을 한 번에 효율적으로 합칩니다.
    updated_array = np.concatenate((target_array, *new_data_list))

    if len(updated_array) > max_len:
        return updated_array[-max_len:]
    return updated_array


def array_modifier(data_arr):
    modified_array = [0] * len(data_arr)

    i = 0
    while i < len(data_arr):
        if data_arr[i] != 0:
            group_start = i
            while i < len(data_arr) and data_arr[i] != 0:
                i += 1
            group = data_arr[group_start:i]
            max_value = max(group)

            if max_value >= 5:
                max_index = np.argmax(group)
                modified_array[group_start + max_index] = 1
                for j in range(group_start, i):
                    if j != group_start + max_index:
                        modified_array[j] = 0
            else:
                for j in range(group_start, i):
                    modified_array[j] = 0
        else:
            i += 1

    return modified_array


def co_determination(heart_arr, ecmo_arr):
    no_bpm = 0

    heart_indices = [i for i, val in enumerate(heart_arr) if val == 1]
    ecmo_indices = [i for i, val in enumerate(ecmo_arr) if val == 1]

    if len(heart_indices) < 2:
        return no_bpm, 7  # stay
    if len(ecmo_indices) < 2:
        return no_bpm, 7  # stay

    heart_length = heart_indices[-1] - heart_indices[-2]
    heart_ecmo_length_over = ecmo_indices[-1] - heart_indices[-1]
    heart_ecmo_length = ecmo_indices[-1] - heart_indices[-2]

    h_bpm = int(60 / (heart_length * 0.016))
    # print('heart_arr:', *heart_arr)
    # print('ecmo_arr:', *ecmo_arr)
    # print('index_0:', heart_indices[0])
    # print('index_-1:', heart_indices[-1])
    # print('index_-2:', heart_indices[-2])
    # print('index_n:', heart_indices)
    # print(h_bpm)
    # print('heart_len:', len(heart_arr))
    # print("heart length", heart_length)
    # print("heart_ECMO length", heart_ecmo_length)

    if heart_indices[-1] == len(heart_arr) - 1:

        if h_bpm <= 50:

            h_bpm = int(60 / (heart_length * 0.0162))

            if 0 <= heart_ecmo_length < heart_length * 0.3:
                return h_bpm, 2  # lag
            elif heart_length * 0.3 <= heart_ecmo_length < heart_length * 0.4:
                return h_bpm, 0  # stay
            elif heart_length * 0.5 <= heart_ecmo_length < heart_length:
                return h_bpm, 1  # lead
            else:
                return h_bpm, 0

        if 50 < h_bpm <= 100:

            h_bpm = int(60 / (heart_length * 0.0163))

            if 0 <= heart_ecmo_length < heart_length * 0.3:
                return h_bpm, 2  # lag
            elif heart_length * 0.3 <= heart_ecmo_length < heart_length * 0.45:
                return h_bpm, 0  # stay
            elif heart_length * 0.45 <= heart_ecmo_length < heart_length:
                return h_bpm, 1  # lead
            else:
                return h_bpm, 0

        # elif heart_length < heart_ecmo_length:
        #     if heart_ecmo_length_over < heart_length * 0.3:
        #         return h_bpm, 1  # lead
        #     elif heart_length * 0.3 <= heart_ecmo_length_over <= heart_length * 0.6:
        #         return h_bpm, 0  # stay
        #     elif heart_length * 0.6 < heart_ecmo_length_over <= heart_length:
        #         return h_bpm, 2  # lag

        elif ecmo_indices[-1] < heart_indices[-2]:
            return h_bpm, 2  # lag

    else:
        return no_bpm, 7
    return no_bpm, 7  # stay


class CounterPulsation(mp.Process):

    def __init__(self, save_spo2_queue, save_outp_h_spo2_queue, save_outp_e_spo2_queue, save_ibp_queue, save_sac1_queue,
                 save_sac2_queue, save_outp_h_queue, save_outp_e_queue, save_flow_queue, save_preload_queue,
                 save_afterload_queue, save_upsample_spo2_queue, save_upsample_ibp_queue, save_upsample_p1_queue,
                 save_upsample_p2_queue, save_upsample_flow_queue, save_upsample_preload_queue,
                 save_upsample_afterload_queue, semaphore3):
        mp.Process.__init__(self)

        self.save_outp_h_spo2_queue = save_outp_h_spo2_queue
        self.save_outp_e_spo2_queue = save_outp_e_spo2_queue
        self.save_outp_h_queue = save_outp_h_queue
        self.save_outp_e_queue = save_outp_e_queue

        self.save_spo2_queue = save_spo2_queue
        self.save_ibp_queue = save_ibp_queue
        self.save_sac1_queue = save_sac1_queue
        self.save_sac2_queue = save_sac2_queue
        self.save_flow_queue = save_flow_queue
        self.save_preload_queue = save_preload_queue
        self.save_afterload_queue = save_afterload_queue

        self.save_upsample_spo2_queue = save_upsample_spo2_queue
        self.save_upsample_ibp_queue = save_upsample_ibp_queue
        self.save_upsample_p1_queue = save_upsample_p1_queue
        self.save_upsample_p2_queue = save_upsample_p2_queue
        self.save_upsample_flow_queue = save_upsample_flow_queue
        self.save_upsample_preload_queue = save_upsample_preload_queue
        self.save_upsample_afterload_queue = save_upsample_afterload_queue

        self.get_outp_h_spo2_data = 0
        self.get_outp_e_spo2_data = 0
        self.get_outp_h_data = 0
        self.get_outp_e_data = 0

        self.get_spo2_diff_data = 0
        self.get_ibp_diff_data = 0
        self.get_sac1_data = 0
        self.get_sac2_data = 0
        self.get_flow_data = 0
        self.get_preload_data = 0
        self.get_afterload_data = 0

        self.get_upsample_spo2_diff_data = 0
        self.get_upsample_ibp_diff_data = 0
        self.get_upsample_sac1_data = 0
        self.get_upsample_sac2_data = 0
        self.get_upsample_flow_data = 0
        self.get_upsample_preload_data = 0
        self.get_upsample_afterload_data = 0

        self.get_outp_h_spo2 = np.array([])
        self.get_outp_e_spo2 = np.array([])
        self.get_outp_h = np.array([])
        self.get_outp_e = np.array([])

        self.get_spo2_diff = np.array([])
        self.get_ibp_diff = np.array([])
        self.get_sac1 = np.array([])
        self.get_sac2 = np.array([])
        self.get_flow = np.array([])
        self.get_preload = np.array([])
        self.get_afterload = np.array([])

        self.get_upsample_spo2_diff = np.array([])
        self.get_upsample_ibp_diff = np.array([])
        self.get_upsample_sac1 = np.array([])
        self.get_upsample_sac2 = np.array(([]))
        self.get_upsample_flow = np.array([])
        self.get_upsample_preload = np.array([])
        self.get_upsample_afterload = np.array([])

        self.save_outp_h_spo2 = np.array([])
        self.save_outp_e_spo2 = np.array([])
        self.save_outp_h = np.array([])
        self.save_outp_e = np.array([])

        self.save_spo2 = np.array([])
        self.save_diff = np.array([])
        self.save_sac1 = np.array([])
        self.save_sac2 = np.array([])
        self.save_flow = np.array([])
        self.save_preload = np.array([])
        self.save_afterload = np.array([])

        self.bpm = 0
        self.bpm_old = 0
        self.bpm_arr = np.array([])
        self.status = 0
        self.status_old = 0

        self.over_bpm = 0

        self.save_bpm = np.array([])
        self.save_status = np.array([])

        self.data_save_flag = 0
        self.duplicate_flag = 0

        self.irregular_count = 0
        self.lag_count = 0
        self.lead_count = 0
        self.stay_count = 0
        self.data_save_count = 0

        self.bpm_stay = 0

        self.determination_loop = 0

        self.sema3 = semaphore3

    def run(self):

        py_serial = serial.Serial(port='COM11', baudrate=921600, timeout=0)

        outp_h_spo2_queue = Queue()
        outp_e_spo2_queue = Queue()
        outp_h_queue = Queue()
        outp_e_queue = Queue()

        spo2_queue = Queue()
        ibp_queue = Queue()
        sac1_queue = Queue()
        sac2_queue = Queue()
        flow_queue = Queue()
        preload_queue = Queue()
        afterload_queue = Queue()

        self.update_counter = 0

        send_bpm = Queue()
        send_status = Queue()
        send_hex_bpm_tens = Queue()
        send_hex_bpm_units = Queue()
        send_hex_status = Queue()
        send_crc = Queue()

        print('Determination Start')

        def determination():
            # try:
            while True:

                if not self.save_ibp_queue.empty():

                    self.get_outp_h_spo2_data = np.array([self.save_outp_h_spo2_queue.get()])
                    self.get_outp_e_spo2_data = np.array([self.save_outp_e_spo2_queue.get()])

                    self.get_spo2_diff_data = np.array([self.save_spo2_queue.get()])
                    self.get_ibp_diff_data = np.array([self.save_ibp_queue.get()])
                    self.get_sac1_data = np.array([self.save_sac1_queue.get()])
                    self.get_sac2_data = np.array([self.save_sac2_queue.get()])
                    self.get_outp_h_data = np.array([self.save_outp_h_queue.get()])
                    self.get_outp_e_data = np.array([self.save_outp_e_queue.get()])
                    self.get_flow_data = np.array([self.save_flow_queue.get()])
                    self.get_preload_data = np.array([self.save_preload_queue.get()])
                    self.get_afterload_data = np.array([self.save_afterload_queue.get()])

                    upsample_spo2_chunks = []
                    upsample_ibp_chunks = []
                    upsample_sac1_chunks = []
                    upsample_sac2_chunks = []
                    upsample_flow_chunks = []
                    upsample_preload_chunks = []
                    upsample_afterload_chunks = []

                    for _ in range(3):
                        upsample_spo2_chunks.append(np.array([self.save_upsample_spo2_queue.get()]))
                        upsample_ibp_chunks.append(np.array([self.save_upsample_ibp_queue.get()]))
                        upsample_sac1_chunks.append(np.array([self.save_upsample_p1_queue.get()]))
                        upsample_sac2_chunks.append(np.array([self.save_upsample_p2_queue.get()]))
                        upsample_flow_chunks.append(np.array([self.save_upsample_flow_queue.get()]))
                        upsample_preload_chunks.append(np.array([self.save_upsample_preload_queue.get()]))
                        upsample_afterload_chunks.append(np.array([self.save_upsample_afterload_queue.get()]))


                    self.get_outp_h_spo2 = alternating_update_array(self.get_outp_h_spo2,
                                                                    upsample_spo2_chunks + [self.get_outp_h_spo2_data])
                    self.get_outp_e_spo2 = alternating_update_array(self.get_outp_e_spo2,
                                                                    upsample_spo2_chunks + [self.get_outp_e_spo2_data])

                    self.get_spo2_diff = alternating_update_array(self.get_spo2_diff,
                                                                  upsample_spo2_chunks + [self.get_spo2_diff_data])
                    self.get_ibp_diff = alternating_update_array(self.get_ibp_diff, upsample_ibp_chunks +
                                                                 [self.get_ibp_diff_data])
                    self.get_sac1 = alternating_update_array(self.get_sac1, upsample_sac1_chunks + [self.get_sac1_data])
                    self.get_sac2 = alternating_update_array(self.get_sac2, upsample_sac2_chunks + [self.get_sac2_data])
                    self.get_outp_h = alternating_update_array(self.get_outp_h, upsample_ibp_chunks +
                                                               [self.get_outp_h_data])
                    self.get_outp_e = alternating_update_array(self.get_outp_e, upsample_ibp_chunks +
                                                               [self.get_outp_e_data])
                    self.get_flow = alternating_update_array(self.get_flow, upsample_flow_chunks + [self.get_flow_data])
                    self.get_preload = alternating_update_array(self.get_preload, upsample_preload_chunks +
                                                                [self.get_preload_data])
                    self.get_afterload = alternating_update_array(self.get_afterload,
                                                                  upsample_afterload_chunks + [self.get_afterload_data])

                    # ibp
                    if len(self.get_ibp_diff) >= 1333333300:                                    # test용 large number
                        heart_result = array_modifier(self.get_outp_h)
                        ecmo_result = array_modifier(self.get_outp_e)

                        self.bpm, self.status = co_determination(heart_result, ecmo_result)

                        self.data_save_flag = 0

                        if 40 <= self.bpm <= 100 and self.save_bpm[-1] == '':

                            self.data_save_flag = 1

                            if self.status == 1:
                                self.lead_count += 1
                            else:
                                self.lead_count = 0

                            if self.status == 2:
                                self.lag_count += 1
                            else:
                                self.lag_count = 0

                            if self.status == 0 and self.bpm == self.bpm_old:
                                self.stay_count += 1
                            else:
                                self.stay_count = 0

                            if self.stay_count >= 5:
                                self.bpm_stay = self.bpm

                            self.bpm_arr = np.append(self.bpm_arr, self.bpm)
                            if len(self.bpm_arr) >= 11:
                                self.bpm_arr = self.bpm_arr[1:]
                                # if self.bpm_arr[-1] > int(np.average(self.bpm_arr)) + 3:
                                #     self.bpm_arr[-1] = int(np.average(self.bpm_arr))
                            self.bpm = int(np.average(self.bpm_arr))

                            if abs(self.bpm_arr[-1] - int(np.average(self.bpm_arr))) > 15:
                                self.bpm_arr[-1] = int(np.average(self.bpm_arr))
                            # self.bpm = int((self.bpm + self.bpm_old) / 2)

                            # if abs(self.bpm_arr[-1] - int(np.average(self.bpm_arr))) > 5:
                            #     self.over_bpm += 1
                            # else:
                            #     self.over_bpm = 0
                            #
                            # if self.over_bpm < 3:
                            #     if (self.status == 2 or self.status_old == 2) and self.bpm > self.bpm_old:
                            #         self.bpm_arr[-1] = self.bpm_old
                            #         self.bpm = self.bpm_old
                            #         print('bpm preserved by lag', self.bpm, self.bpm_old)
                            #
                            #     if (self.status == 1 or self.status_old == 1) and self.bpm < self.bpm_old:
                            #         self.bpm_arr[-1] = self.bpm_old
                            #         self.bpm = self.bpm_old
                            #         print('bpm preserved by lead', self.bpm, self.bpm_old)
                            # else:
                            #     pass

                            if self.status == 0:
                                self.bpm = self.bpm_old

                            if self.status_old == 0 and self.status == 2 and self.bpm > self.bpm_old:
                                self.bpm = self.bpm_old
                            if self.status_old == 0 and self.status == 1 and self.bpm < self.bpm_old:
                                self.bpm = self.bpm_old

                            # if len(self.bpm_arr) > 9:
                            #     if self.status == 1:
                            #         if self.bpm < self.bpm_old:
                            #             self.bpm = self.bpm_old
                            #         if self.bpm >= self.bpm_old + 2:
                            #             self.bpm = self.bpm_old
                            #
                            #     if self.status == 2:
                            #         if self.bpm > self.bpm_old:
                            #             self.bpm = self.bpm_old
                            #         if self.bpm <= self.bpm_old - 2:
                            #             self.bpm = self.bpm_old

                            if self.stay_count >= 5:
                                self.bpm_arr[-1] = self.bpm_stay
                                self.bpm = self.bpm_stay

                            # if self.lag_count > 10:
                            #     self.bpm = self.bpm - 1
                            #     self.lag_count = 0
                            # if self.lead_count > 10:
                            #     self.bpm = self.bpm + 1
                            #     self.lead_count = 0

                            print('40~80: ', *self.bpm_arr)
                            print('40~80: ', self.bpm, self.bpm_old, self.status)
                            print('')

                            send_bpm.put(self.bpm)
                            send_status.put(self.status)

                            self.bpm_old = self.bpm
                            self.status_old = self.status

                        else:
                            pass

                        outp_h_spo2_queue.put(self.get_outp_h_spo2_data)
                        outp_e_spo2_queue.put(self.get_outp_e_spo2_data)
                        outp_h_queue.put(self.get_outp_h_data)
                        outp_e_queue.put(self.get_outp_e_data)

                        spo2_queue.put(self.get_spo2_diff_data)
                        ibp_queue.put(self.get_ibp_diff_data)
                        sac1_queue.put(self.get_sac1_data)
                        sac2_queue.put(self.get_sac2_data)
                        flow_queue.put(self.get_flow_data)
                        preload_queue.put(self.get_preload_data)
                        afterload_queue.put(self.get_afterload_data)

        # except KeyboardInterrupt:
        #     print("Determination Process interrupted by user.")
        # except Exception as error:
        #     print("An error occurred in Determination Process:", str(error))

        def save_file():
            try:
                while True:

                    outp_h_spo2_data = outp_h_spo2_queue.get()
                    outp_e_spo2_data = outp_e_spo2_queue.get()
                    outp_h_data = outp_h_queue.get()
                    outp_e_data = outp_e_queue.get()

                    spo2_data = spo2_queue.get()
                    ibp_data = ibp_queue.get()
                    sac1_data = sac1_queue.get()
                    sac2_data = sac2_queue.get()
                    flow_data = flow_queue.get()
                    preload_data = preload_queue.get()
                    afterload_data = afterload_queue.get()

                    if len(self.get_ibp_diff) > 0:

                        self.save_spo2 = np.append(self.save_spo2, spo2_data)
                        # print("spo2:", *self.save_spo2)
                        self.save_outp_h_spo2 = np.append(self.save_outp_h_spo2, outp_h_spo2_data)
                        self.save_outp_e_spo2 = np.append(self.save_outp_e_spo2, outp_e_spo2_data)

                        self.save_diff = np.append(self.save_diff, ibp_data)
                        # print("diff:", *self.save_diff)
                        self.save_sac1 = np.append(self.save_sac1, sac1_data)
                        self.save_sac2 = np.append(self.save_sac2, sac2_data)
                        self.save_outp_h = np.append(self.save_outp_h, outp_h_data)
                        self.save_outp_e = np.append(self.save_outp_e, outp_e_data)
                        self.save_flow = np.append(self.save_flow, flow_data)
                        self.save_preload = np.append(self.save_preload, preload_data)
                        self.save_afterload = np.append(self.save_afterload, afterload_data)

                        if self.data_save_flag == 0:
                            self.save_bpm = np.append(self.save_bpm, '')
                            self.save_status = np.append(self.save_status, '')
                        elif self.data_save_flag == 1:
                            self.save_bpm = np.append(self.save_bpm, self.bpm)
                            self.save_status = np.append(self.save_status, self.status)

                        if self.determination_loop == 3000:
                            labels = ["spo2", "est_h_spo2", "est_e_spo2", "preload", "ibp_raw", "ibp_diff", "sac1",
                                      "sac2", "flow", "est_h_ibp", "est_e_ibp", "bpm", "status"]

                            save_data = np.vstack((self.save_spo2, self.save_outp_h_spo2, self.save_outp_e_spo2,
                                                   self.save_preload, self.save_afterload, self.save_diff,
                                                   self.save_sac1, self.save_sac2, self.save_flow,
                                                   self.save_outp_h, self.save_outp_e, self.save_bpm,
                                                   self.save_status)).T

                            labels_array = np.array(labels).reshape(1, len(labels))
                            save_data_with_labels = np.concatenate((labels_array, save_data), axis=0)

                            np.savetxt(
                                r'C:\Users\user\Desktop\training_file\results\25_May\CP_v2\test\ecmo_ai_apply_250513_a%s.csv'
                                % self.data_save_count, save_data_with_labels, fmt='%s', delimiter=",")

                            self.data_save_count += 1

                            self.save_spo2 = np.array([])
                            self.save_outp_h_spo2 = np.array([])
                            self.save_outp_e_spo2 = np.array([])

                            self.save_diff = np.array([])
                            self.save_sac1 = np.array([])
                            self.save_sac2 = np.array([])
                            self.save_outp_h = np.array([])
                            self.save_outp_e = np.array([])
                            self.save_bpm = np.array([])
                            self.save_status = np.array([])
                            self.save_flow = np.array([])
                            self.save_preload = np.array([])
                            self.save_afterload = np.array([])

                            self.determination_loop = 0

                            print('data saved')

                    self.determination_loop += 1

            except KeyboardInterrupt:
                print("save_file Process interrupted by user.")
            except Exception as error:
                print("An error occurred in save_file Process:", str(error))

        def data_packet_processing():
            try:

                mapping = {
                    0: '30',
                    1: '31',
                    2: '32',
                    3: '33',
                    4: '34',
                    5: '35',
                    6: '36',
                    7: '37',
                    8: '38',
                    9: '39'
                }

                while True:
                    receive_bpm = send_bpm.get()
                    receive_status = send_status.get()

                    bpm_str = str(receive_bpm)

                    bpm_digit_1 = int(bpm_str[0])  # 10의 자리
                    bpm_digit_2 = int(bpm_str[1])  # 1의 자리

                    bpm_hex_tens = mapping.get(bpm_digit_1, 4)
                    bpm_hex_units = mapping.get(bpm_digit_2, 0)
                    status_hex = mapping.get(receive_status, 0)

                    hex_addition = int('80', 16) + int(bpm_hex_tens, 16) + int(bpm_hex_units, 16) + int(status_hex, 16)
                    hex_and = hex_addition & int('7f', 16)

                    crc = hex(hex_and)[2:].upper()  # 0x 삭제 후 대문자 변환

                    send_hex_bpm_tens.put(bpm_hex_tens)
                    send_hex_bpm_units.put(bpm_hex_units)
                    send_hex_status.put(status_hex)
                    send_crc.put(crc)

            except KeyboardInterrupt:
                print("Determination Process interrupted by user.")
            except Exception as error:
                print("An error occurred in Determination Process:", str(error))

        def packet_send():
            try:
                while True:
                    get_hex_bpm_tens = send_hex_bpm_tens.get()
                    get_hex_bpm_units = send_hex_bpm_units.get()
                    get_hex_status = send_hex_status.get()
                    get_crc = send_crc.get()

                    # print(get_hex_bpm_tens, get_hex_bpm_units, get_hex_status, get_crc)

                    hex_to_bytes = bytes([0x80]) + bytes.fromhex(get_hex_bpm_tens) + \
                                   bytes.fromhex(get_hex_bpm_units) + bytes.fromhex(get_hex_status) + \
                                   bytes.fromhex(get_crc) + bytes([0xff])

                    py_serial.write(hex_to_bytes)

            except KeyboardInterrupt:
                print("Packet Send Process interrupted by user.")
            except Exception as e:
                print("An error occurred in Packet Send Process:", str(e))

        thread1 = threading.Thread(target=determination)
        thread2 = threading.Thread(target=save_file)
        thread3 = threading.Thread(target=data_packet_processing)
        thread4 = threading.Thread(target=packet_send)

        thread1.start()
        thread2.start()
        thread3.start()
        thread4.start()

        thread1.join()
        thread2.join()
        thread3.join()
        thread4.join()
