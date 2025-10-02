import multiprocessing as mp
import threading
import serial
import numpy as np
from queue import Queue
from Function import alternating_update_array, array_modifier, co_determination, calculate_bpm_adjustment

from tensorflow import keras
from collections import deque

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

        # Debugging 용 Worker0의 62.5Hz IBP 배열
        self.get_ibp_worker0 = np.array([])

        self.bpm = 0
        self.bpm_old = 0
        self.bpm_arr = np.array([])
        self.status = 0
        self.status_old = 0
        self.cal_success = True

        self.bpm_flag = 0

        self.lag_count = 0
        self.lead_count = 0
        self.stay_count = 0
        self.data_save_count = 0

        self.bpm_stay = 0

        self.determination_loop = 0

        # self.sema3 = semaphore3

        self.last_processed_heart_peak_pos = -1
        self.previous_phase_error = 0.0

    def run(self):

        py_serial = serial.Serial(port='COM7', baudrate=921600, timeout=0)  # 송신할 comport 및 baudrate 체크

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

                ### [1. AI 모델 로딩 및 준비 (최초 1회 실행)] ###
                try:
                    # 훈련된 AI 모델을 불러옵니다. (경로는 실제 파일 위치에 맞게 수정)
                    model_path = r'C:\Users\user\Desktop\training_file\results\25_Sep\data\modified\bpm_control_model_250926.h5'
                    control_model = keras.models.load_model(model_path)
                    print("BPM control AI model loaded successfully.")
                except Exception as e:
                    print(f"Error loading AI model: {e}")
                    control_model = None  # 모델 로딩 실패 시 None으로 설정

                # 모델의 입력 시퀀스 길이에 맞게 설정 (예: 5)
                SEQ_LENGTH = 5
                # (ratio, h_interval_sec, e_interval_sec) 튜플 5개를 저장할 공간
                # (0.0, 1.0, 1.0) -> ratio=0, h_interval=1초, e_interval=1초 (60BPM)로 초기화
                recent_events = deque([(0.5, 0.5)] * SEQ_LENGTH, maxlen=SEQ_LENGTH)

                WINDOW_SIZE = 500  # self.get_outp_h의 최대 크기
                NEW_PEAK_THRESHOLD = 3

                # 새로운 delay_ratio를 계산하는 헬퍼 함수
                def calculate_latest_delay_ratio(h_peaks, e_peaks):
                    if len(h_peaks) < 2:
                        return None  # 계산 불가

                    h_prev, h_curr = h_peaks[-2], h_peaks[-1]
                    interval = h_curr - h_prev
                    if interval == 0:
                        return None

                    # 해당 주기 내의 ECMO 펄스 찾기
                    for e_peak in reversed(e_peaks):
                        if h_prev < e_peak < h_curr:
                            return (e_peak - h_prev) / interval
                    return None  # ECMO 펄스를 못찾음

                while True:

                    data_frame = [[] for _ in range(11)]

                    for worker_idx in range(4):  # 각 worker 에서 결과값 수신
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

                    # 4분할 데이터 합친 후 길이 조절 (spo2, ibp, pump1, pump2, 심장 및 ECMO 펄스 결과 값)
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

                    # Debugging 용 ibp print
                    # ibp_from_worker0 = ibp_diff_block[0]
                    # self.get_ibp_worker0 = alternating_update_array(self.get_ibp_worker0, [ibp_from_worker0], max_len=125)
                    # np.set_printoptions(threshold=np.inf, linewidth=np.inf)
                    # heart_result = array_modifier(self.get_outp_h)
                    # print(f"62.5Hz (W0): {self.get_ibp_worker0}")
                    # print(f"250Hz ibp_diff: {self.get_ibp_diff}")
                    # print(f"250Hz ibp: {self.get_afterload}")
                    # print(f"250Hz heart: {self.get_outp_h}")
                    # print(f"250Hz heart: {heart_result}")
                    # print('')

                    # ibp
                    if len(self.get_ibp_diff) >= 500:

                        heart_peak_positions = array_modifier(self.get_outp_h)  # 심장 펄스 위치 값
                        ecmo_peak_positions = array_modifier(self.get_outp_e)  # ECMO 펄스 위치 값

                        #########################기존 PLL 로직#########################
                        # 심장, ECMO 펄스 위치 확인 후 ECMO 동작 결정
                        # if len(heart_peak_positions) > 0:
                        #     if (len(self.get_outp_h) - heart_peak_positions[-1]) < 4:
                        #         self.bpm, self.status, self.cal_success = co_determination(heart_peak_positions,
                        #                                                                    ecmo_peak_positions)
                        #
                        # if self.bpm > 0 and self.cal_success:
                        #
                        #     if self.status == 1:
                        #         self.lead_count += 1
                        #     else:
                        #         self.lead_count = 0
                        #
                        #     if self.status == 2:
                        #         self.lag_count += 1
                        #     else:
                        #         self.lag_count = 0
                        #
                        #     if self.status == 0 and self.bpm == self.bpm_old:
                        #         self.stay_count += 1
                        #     else:
                        #         self.stay_count = 0
                        #
                        #     if self.stay_count >= 5:
                        #         self.bpm_stay = self.bpm
                        #
                        #     # 급격한 bpm 변화(오류) 영향 방지
                        #     self.bpm_arr = np.append(self.bpm_arr, self.bpm)
                        #     if len(self.bpm_arr) >= 11:
                        #         self.bpm_arr = self.bpm_arr[1:]
                        #         # if self.bpm_arr[-1] > int(np.average(self.bpm_arr)) + 3:
                        #         #     self.bpm_arr[-1] = int(np.average(self.bpm_arr))
                        #     self.bpm = int(np.average(self.bpm_arr))
                        #
                        #     if abs(self.bpm_arr[-1] - int(np.average(self.bpm_arr))) > 15:
                        #         self.bpm_arr[-1] = int(np.average(self.bpm_arr))
                        #
                        #     if self.status == 0:
                        #         self.bpm = self.bpm_old
                        #
                        #     if self.status_old == 0 and self.status == 2 and self.bpm > self.bpm_old:
                        #         self.bpm = self.bpm_old
                        #     if self.status_old == 0 and self.status == 1 and self.bpm < self.bpm_old:
                        #         self.bpm = self.bpm_old
                        #
                        #     if self.stay_count >= 5:
                        #         self.bpm_arr[-1] = self.bpm_stay
                        #         self.bpm = self.bpm_stay
                        #
                        #     # bpm 출력 디버깅
                        #     print('Determination, 0~100: ', *self.bpm_arr)
                        #     print('Determination, 0~100: ', self.bpm, self.bpm_old, self.status)
                        #     print('')
                        #
                        #     send_bpm.put(self.bpm)
                        #     send_status.put(self.status)
                        #
                        #     self.bpm_old = self.bpm
                        #     self.status_old = self.status
                        #
                        # else:
                        #     pass

                        ######################새로운 로직###########################
                        # ibp
                        # if len(self.get_ibp_diff) >= 500:
                        #
                        #     heart_peak_positions = array_modifier(self.get_outp_h)
                        #     ecmo_peak_positions = array_modifier(self.get_outp_e)
                        #
                        #     # --- 해결책: 변수들의 기본값을 미리 설정 ---
                        #     # 계산이 실패할 것을 대비하여 기본 상태를 설정합니다.
                        #     self.cal_success = False
                        #     measured_h_bpm = self.bpm_old  # 기본 BPM은 이전 값으로
                        #     bpm_adjustment = 0  # 조정값은 0으로
                        #
                        #     # 조건이 맞을 경우에만 위 변수들의 값을 덮어쓰도록 합니다.
                        #     if len(heart_peak_positions) > 0:
                        #         if (len(self.get_outp_h) - heart_peak_positions[-1]) < 4:
                        #             measured_h_bpm, bpm_adjustment, self.cal_success = calculate_bpm_adjustment(
                        #                 heart_peak_positions,
                        #                 ecmo_peak_positions,
                        #                 self.bpm_old
                        #             )
                        #
                        #     # 이제 cal_success가 True일 때만 measured_h_bpm이 안전하게 사용됩니다.
                        #     if self.cal_success:
                        #         self.bpm = measured_h_bpm + bpm_adjustment
                        #         self.bpm_arr = np.append(self.bpm_arr, self.bpm)
                        #         if len(self.bpm_arr) >= 5:
                        #             self.bpm_arr = self.bpm_arr[1:]
                        #         self.bpm = int(np.average(self.bpm_arr))
                        #         self.bpm = np.clip(self.bpm, 40, 100)
                        #
                        #         print(
                        #             f"Heart BPM: {measured_h_bpm}, Adjustment: {bpm_adjustment}, Target ECMO BPM: {self.bpm}")
                        #         print(f"BPM Array: {self.bpm_arr}")
                        #         print('')
                        #
                        #         send_bpm.put(self.bpm)
                        #         send_status.put(bpm_adjustment)
                        #
                        #     else:
                        #         # 계산 실패 시 (cal_success가 False일 때), 이전 BPM을 그대로 사용합니다.
                        #         self.bpm = self.bpm_old
                        #         send_bpm.put(self.bpm)
                        #         send_status.put(0)
                        #
                        #     # 루프 마지막에 bpm_old 값을 현재 bpm으로 업데이트합니다.
                        #     self.bpm_old = self.bpm

                        #####################새로운 DNN 학습모델 적용#########################
                        if len(heart_peak_positions) > 0 and (
                                WINDOW_SIZE - heart_peak_positions[-1]) < NEW_PEAK_THRESHOLD:

                            if len(heart_peak_positions) >= 2:
                                h_prev, h_curr = heart_peak_positions[-2], heart_peak_positions[-1]
                                interval_samples = h_curr - h_prev
                                current_heart_bpm = 60 / (
                                            interval_samples * 0.004) if interval_samples > 0 else self.bpm_old

                                BPM_CHANGE_LIMIT = 20

                                if self.bpm_old > 0 and abs(current_heart_bpm - self.bpm_old) > BPM_CHANGE_LIMIT:
                                    print(
                                        f"!! Outlier Detected & Skipped: H-BPM {current_heart_bpm:.1f} is too different from previous BPM {self.bpm_old}.")
                                else:
                                    # --- ⭐ [수정] 10개 입력을 위한 특징(Feature) 계산 ---

                                    # 1. 현재 주기의 ECMO 펄스 찾기
                                    current_ecmo_peak_rel = -1
                                    for e_peak in reversed(ecmo_peak_positions):
                                        if h_prev < e_peak < h_curr:
                                            current_ecmo_peak_rel = e_peak
                                            break

                                    h_e_delay_sec, e_h_delay_sec = None, None
                                    if current_ecmo_peak_rel != -1:
                                        # 2. H-E delay와 E-H delay를 '초' 단위로 계산
                                        h_e_delay_sec = (current_ecmo_peak_rel - h_prev) * 0.004
                                        e_h_delay_sec = (h_curr - current_ecmo_peak_rel) * 0.004

                                    # --- AI 예측 ---
                                    bpm_adjustment = 0

                                    # 두 특징이 모두 유효할 때만 AI 예측 수행
                                    if h_e_delay_sec is not None and e_h_delay_sec is not None and h_e_delay_sec > 0:

                                        # ⭐ 새로운 2-튜플 이벤트를 deque에 추가
                                        new_event = (h_e_delay_sec, e_h_delay_sec)
                                        recent_events.append(new_event)

                                        # ⭐ 5개의 튜플에서 10개짜리 리스트(입력 데이터) 생성
                                        h_e_delays = [event[0] for event in recent_events]
                                        e_h_delays = [event[1] for event in recent_events]
                                        input_data = np.array(h_e_delays + e_h_delays).reshape(1, 10)

                                        prediction = control_model.predict(input_data, verbose=0)
                                        action_index = np.argmax(prediction)

                                        if action_index == 0:
                                            bpm_adjustment = -2  # 강한 감속
                                        elif action_index == 1:
                                            bpm_adjustment = -1  # 약한 감속
                                        elif action_index == 2:
                                            bpm_adjustment = 0  # 유지
                                        elif action_index == 3:
                                            bpm_adjustment = 1  # 약한 가속
                                        elif action_index == 4:
                                            bpm_adjustment = 2  # 강한 가속

                                    # --- 최종 BPM 계산 및 전송 ---
                                    target_ecmo_bpm = int(round(current_heart_bpm + bpm_adjustment))

                                    self.bpm_arr = np.append(self.bpm_arr, target_ecmo_bpm)
                                    if len(self.bpm_arr) >= 5:
                                        self.bpm_arr = self.bpm_arr[1:]
                                    smoothed_bpm = int(np.average(self.bpm_arr))

                                    self.bpm = smoothed_bpm
                                    self.bpm = np.clip(self.bpm, 40, 100)

                                    # ⭐ Print 문 수정
                                    if h_e_delay_sec is not None:
                                        # delay_ratio를 즉석에서 계산하여 참고용으로 출력
                                        ratio_display = h_e_delay_sec / (h_e_delay_sec + e_h_delay_sec)
                                        print(
                                            f"H-BPM:{current_heart_bpm:.1f}, Ratio:{ratio_display:.2f}, AI Adj:{bpm_adjustment} -> Raw:{target_ecmo_bpm}, Smooth:{self.bpm}")
                                    else:
                                        print(
                                            f"H-BPM:{current_heart_bpm:.1f}, Ratio:N/A -> Raw:{target_ecmo_bpm}, Smooth:{self.bpm}")

                                    send_bpm.put(self.bpm)
                                    send_status.put(bpm_adjustment)
                                    self.bpm_old = self.bpm

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

                    if not self.cal_success or self.bpm == 0:
                        self.save_bpm = np.append(self.save_bpm, [''] * 4)
                        self.save_status = np.append(self.save_status, [''] * 4)
                    elif self.cal_success and self.bpm > 0:
                        self.save_bpm = np.append(self.save_bpm, ['', '', '', self.bpm])
                        self.save_status = np.append(self.save_status, ['', '', '', self.status])

                    self.determination_loop += 4

                    # 저장할 데이터 길이 및 정보
                    if self.determination_loop == 2000:
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
                            r'C:\Users\user\Desktop\training_file\results\25_Sep\test\ecmo_ai_apply_250919_a%s'
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
