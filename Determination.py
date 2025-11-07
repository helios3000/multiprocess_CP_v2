import multiprocessing as mp
import threading
import serial
import numpy as np
from queue import Queue
from Function import alternating_update_array, array_modifier, calculate_cardiac_metrics


class CounterPulsation(mp.Process):

    def __init__(self, all_worker_out_queues):
        mp.Process.__init__(self)

        self.worker_queues = [
            all_worker_out_queues[0],
            all_worker_out_queues[1],
            all_worker_out_queues[2],
            all_worker_out_queues[3]
        ]

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
        self.save_bpm = np.array([])
        self.save_status = np.array([])

        self.save_he_delay = np.array([])
        self.save_eh_delay = np.array([])
        self.save_rs_h_prev = np.array([])
        self.save_rs_h_curr = np.array([])
        self.save_rs_e = np.array([])

        self.get_ibp_worker0 = np.array([])

        self.bpm = 0
        self.bpm_old = 0
        self.bpm_arr = np.array([])
        self.status = 0
        self.status_old = 0
        self.cal_success = False

        self.bpm_flag = 0

        self.lag_count = 0
        self.lead_count = 0
        self.stay_count = 0
        self.data_save_count = 0

        self.bpm_stay = 0
        self.determination_loop = 0

        self.bpm_save_index = -1
        self.status_save_index = -1

        self.he_delay = ''
        self.eh_delay = ''
        self.rs_h_prev = ''
        self.rs_h_curr = ''
        self.rs_e = ''
        self.previous_h_curr_rs_value = ''

        self.last_processed_heart_peak_pos = -1
        self.previous_phase_error = 0.0

    def run(self):

        py_serial = serial.Serial(port='COM11', baudrate=921600, timeout=0)  # mock: COM7

        # 스레드 간 통신을 위한 내부 큐
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

        send_bpm = Queue()
        send_status = Queue()
        send_hex_bpm_tens = Queue()
        send_hex_bpm_units = Queue()
        send_hex_status = Queue()
        send_crc = Queue()

        print('Determination Start')

        def determination():

            try:
                while True:

                    data_frame = [[] for _ in range(11)]
                    for worker_idx in range(4):
                        worker_q_list = self.worker_queues[worker_idx]
                        for channel_idx in range(11):
                            data = np.array([worker_q_list[channel_idx].get()])
                            data_frame[channel_idx].append(data)

                    spo2_diff_block = data_frame[0]
                    ibp_diff_block = data_frame[1]
                    sac1_block = data_frame[2]
                    sac2_block = data_frame[3]
                    flow_block = data_frame[4]
                    preload_block = data_frame[5]
                    afterload_block = data_frame[6]
                    outp_h_spo2_block = data_frame[7]
                    outp_e_spo2_block = data_frame[8]
                    outp_h_block = data_frame[9]
                    outp_e_block = data_frame[10]

                    self.get_spo2_diff = alternating_update_array(self.get_spo2_diff, spo2_diff_block)
                    self.get_ibp_diff = alternating_update_array(self.get_ibp_diff, ibp_diff_block)
                    self.get_sac1 = alternating_update_array(self.get_sac1, sac1_block)
                    self.get_sac2 = alternating_update_array(self.get_sac2, sac2_block)
                    self.get_flow = alternating_update_array(self.get_flow, flow_block)
                    self.get_preload = alternating_update_array(self.get_preload, preload_block)
                    self.get_afterload = alternating_update_array(self.get_afterload, afterload_block)
                    self.get_outp_h_spo2 = alternating_update_array(self.get_outp_h_spo2, outp_h_spo2_block)
                    self.get_outp_e_spo2 = alternating_update_array(self.get_outp_e_spo2, outp_e_spo2_block)
                    self.get_outp_h = alternating_update_array(self.get_outp_h, outp_h_block)
                    self.get_outp_e = alternating_update_array(self.get_outp_e, outp_e_block)

                    if len(self.get_ibp_diff) >= 500:

                        self.cal_success = False
                        self.bpm_save_index = -1
                        self.status_save_index = -1
                        self.he_delay = ''
                        self.eh_delay = ''
                        self.rs_h_prev = ''
                        self.rs_h_curr = ''
                        self.rs_e = ''

                        heart_peak_positions = array_modifier(self.get_outp_h, min_distance=80)  # 심장 펄스 위치 값
                        ecmo_peak_positions = array_modifier(self.get_outp_e, min_distance=80)  # ECMO 펄스 위치 값

                        last_peak_pos = heart_peak_positions[-1] if len(heart_peak_positions) > 0 else -1

                        if last_peak_pos >= 496:

                            print(f"heart_peak_positions: {heart_peak_positions}")
                            print(f"ecmo_peak_positions: {ecmo_peak_positions}")
                            print('')

                            (bpm_calc, he_delay_calc, eh_delay_calc,
                             rs_h_prev_calc, rs_h_curr_calc, rs_e_calc,
                             new_prev_h_curr_rs_value, calc_success) = calculate_cardiac_metrics(
                                heart_peak_positions,
                                ecmo_peak_positions,
                                self.previous_h_curr_rs_value,  # 이전 RS '값' 전달
                                self.get_outp_h,  # H 원본 배열 전달
                                self.get_outp_e  # E 원본 배열 전달
                            )

                            if calc_success:
                                self.bpm = bpm_calc
                                self.he_delay = he_delay_calc
                                self.eh_delay = eh_delay_calc
                                self.rs_h_prev = rs_h_prev_calc
                                self.rs_h_curr = rs_h_curr_calc
                                self.rs_e = rs_e_calc
                                # 다음 루프를 위해 새 RS '값' 저장
                                self.previous_h_curr_rs_value = new_prev_h_curr_rs_value

                                self.cal_success = True
                                self.bpm_save_index = int(last_peak_pos) - 496
                                self.status_save_index = int(last_peak_pos) - 496

                                print(f"BPM Calculated: {self.bpm}, Save Slot: {self.bpm_save_index}")
                                print(f"H-E Delay: {self.he_delay}s, E-H Delay: {self.eh_delay}s")
                                print(f"RS (Value): H_prev({self.rs_h_prev}), H_curr({self.rs_h_curr}), E({self.rs_e})")
                                print('')

                        send_bpm.put(self.bpm)
                        send_status.put(self.status)

                        self.bpm_old = self.bpm
                        self.status_old = self.status

                    for i in range(4):
                        outp_h_spo2_queue.put(outp_h_spo2_block[i])
                        outp_e_spo2_queue.put(outp_e_spo2_block[i])
                        outp_h_queue.put(outp_h_block[i])
                        outp_e_queue.put(outp_e_block[i])

                        spo2_queue.put(spo2_diff_block[i])
                        ibp_queue.put(ibp_diff_block[i])
                        sac1_queue.put(sac1_block[i])
                        sac2_queue.put(sac2_block[i])
                        flow_queue.put(flow_block[i])
                        preload_queue.put(preload_block[i])
                        afterload_queue.put(afterload_block[i])

            except KeyboardInterrupt:
                print("Determination Process interrupted by user.")
            except Exception as error:
                print("An error occurred in Determination Process:", str(error))

        def save_file():
            try:
                while True:

                    for _ in range(4):
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

                        self.save_spo2 = np.append(self.save_spo2, spo2_data)
                        self.save_outp_h_spo2 = np.append(self.save_outp_h_spo2, outp_h_spo2_data)
                        self.save_outp_e_spo2 = np.append(self.save_outp_e_spo2, outp_e_spo2_data)
                        self.save_diff = np.append(self.save_diff, ibp_data)
                        self.save_sac1 = np.append(self.save_sac1, sac1_data)
                        self.save_sac2 = np.append(self.save_sac2, sac2_data)
                        self.save_outp_h = np.append(self.save_outp_h, outp_h_data)
                        self.save_outp_e = np.append(self.save_outp_e, outp_e_data)
                        self.save_flow = np.append(self.save_flow, flow_data)
                        self.save_preload = np.append(self.save_preload, preload_data)
                        self.save_afterload = np.append(self.save_afterload, afterload_data)

                    if self.cal_success and (0 <= self.bpm_save_index <= 3):

                        bpm_array = ['', '', '', '']
                        status_array = ['', '', '', '']
                        he_delay_array = ['', '', '', '']
                        eh_delay_array = ['', '', '', '']
                        rs_h_prev_array = ['', '', '', '']
                        rs_h_curr_array = ['', '', '', '']
                        rs_e_array = ['', '', '', '']

                        bpm_array[self.bpm_save_index] = self.bpm
                        status_array[self.status_save_index] = self.status
                        he_delay_array[self.bpm_save_index] = self.he_delay
                        eh_delay_array[self.bpm_save_index] = self.eh_delay
                        rs_h_prev_array[self.bpm_save_index] = self.rs_h_prev
                        rs_h_curr_array[self.bpm_save_index] = self.rs_h_curr
                        rs_e_array[self.bpm_save_index] = self.rs_e

                        self.save_bpm = np.append(self.save_bpm, bpm_array)
                        self.save_status = np.append(self.save_status, status_array)
                        self.save_he_delay = np.append(self.save_he_delay, he_delay_array)
                        self.save_eh_delay = np.append(self.save_eh_delay, eh_delay_array)
                        self.save_rs_h_prev = np.append(self.save_rs_h_prev, rs_h_prev_array)
                        self.save_rs_h_curr = np.append(self.save_rs_h_curr, rs_h_curr_array)
                        self.save_rs_e = np.append(self.save_rs_e, rs_e_array)

                    else:
                        self.save_bpm = np.append(self.save_bpm, [''] * 4)
                        self.save_status = np.append(self.save_status, [''] * 4)
                        self.save_he_delay = np.append(self.save_he_delay, [''] * 4)
                        self.save_eh_delay = np.append(self.save_eh_delay, [''] * 4)
                        self.save_rs_h_prev = np.append(self.save_rs_h_prev, [''] * 4)
                        self.save_rs_h_curr = np.append(self.save_rs_h_curr, [''] * 4)
                        self.save_rs_e = np.append(self.save_rs_e, [''] * 4)

                    self.determination_loop += 4

                    if self.determination_loop == 5000:
                        labels = ["spo2", "est_h_spo2", "est_e_spo2", "preload", "ibp_raw", "ibp_diff", "sac1",
                                  "sac2", "flow", "est_h_ibp", "est_e_ibp", "bpm", "status",
                                  "he_delay", "eh_delay", "rs_h_prev", "rs_h_curr", "rs_e"]

                        save_data = np.vstack((self.save_spo2, self.save_outp_h_spo2, self.save_outp_e_spo2,
                                               self.save_preload, self.save_afterload, self.save_diff,
                                               self.save_sac1, self.save_sac2, self.save_flow,
                                               self.save_outp_h, self.save_outp_e, self.save_bpm,
                                               self.save_status,
                                               self.save_he_delay, self.save_eh_delay,
                                               self.save_rs_h_prev, self.save_rs_h_curr, self.save_rs_e
                                               )).T


                        labels_array = np.array(labels).reshape(1, len(labels))
                        save_data_with_labels = np.concatenate((labels_array, save_data), axis=0)

                        np.savetxt(
                            r'C:\Users\user\Desktop\training_file\results\25_Sep\test\ecmo_ai_apply_251105_a%s'
                            r'.csv'
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
                        self.save_he_delay = np.array([])
                        self.save_eh_delay = np.array([])
                        self.save_rs_h_prev = np.array([])
                        self.save_rs_h_curr = np.array([])
                        self.save_rs_e = np.array([])


                        self.determination_loop = 0

                        print('data saved')

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

                    bpm_str = str(receive_bpm).zfill(2)

                    bpm_digit_1 = int(bpm_str[0])  # 10의 자리
                    bpm_digit_2 = int(bpm_str[1])  # 1의 자리

                    bpm_hex_tens = mapping.get(bpm_digit_1, '30')
                    bpm_hex_units = mapping.get(bpm_digit_2, '30')
                    status_hex = mapping.get(receive_status, '30')

                    # bpm 값 송신을 위해 hex로 변환
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
            # --- [수정 없음] 원본 코드와 동일 ---
            try:
                while True:
                    get_hex_bpm_tens = send_hex_bpm_tens.get()
                    get_hex_bpm_units = send_hex_bpm_units.get()
                    get_hex_status = send_hex_status.get()
                    get_crc = send_crc.get()

                    # print(get_hex_bpm_tens, get_hex_bpm_units, get_hex_status, get_crc)

                    # hex로 변환된 bpm 송신
                    hex_to_bytes = bytes([0x80]) + bytes.fromhex(get_hex_bpm_tens) + \
                                   bytes.fromhex(get_hex_bpm_units) + bytes.fromhex(get_hex_status) + \
                                   bytes.fromhex(get_crc) + bytes([0xff])

                    py_serial.write(hex_to_bytes)

            except KeyboardInterrupt:
                print("Packet Send Process interrupted by user.")
            except Exception as e:
                print("An error occurred in Packet Send Process:", str(e))

        # --- [수정 없음] 스레드 시작 및 실행 ---
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
