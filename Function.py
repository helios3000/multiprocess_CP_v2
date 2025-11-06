import time
import psutil
import numpy as np
from scipy.signal import find_peaks


def monitor_memory(process_map, interval=5):

    print("--- Memory monitor started, waiting for processes to initialize... ---")
    time.sleep(3)

    while True:
        total_rss = 0
        log_message = "--- Memory Usage (PID: Usage) ---\n"

        for name, p_obj in process_map.items():
            try:
                if p_obj.is_alive():
                    ps_p = psutil.Process(p_obj.pid)
                    rss_mb = ps_p.memory_info().rss / (1024 * 1024)
                    log_message += f"{name}({p_obj.pid}): {rss_mb:.1f} MB | "
                    total_rss += rss_mb
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                log_message += f"{name}: N/A | "

        log_message += f"\n>>> Total Subprocess Memory: {total_rss:.1f} MB"
        log_message += "\n------------------------------------"
        print(log_message)

        time.sleep(interval)


def moving_average(data, window_size=5):
    filtered_data = np.sum(data[-window_size:]) / window_size
    return filtered_data


def derivation(current_val, previous_val):
    if previous_val is None:
        return 0.0
    else:
        return (current_val - previous_val) * 0.004


def dnn(x, h1_w, h1_b, h2_w, h2_b, o_w, o_b):

    z1 = np.matmul(x, h1_w) + h1_b
    z1 = np.array(z1.reshape(z1.shape[0] * z1.shape[1]))
    z1 = np.maximum(z1, 0)

    z2 = np.matmul(z1, h2_w) + h2_b
    z2 = np.array(z2.reshape(z2.shape[0] * z2.shape[1]))
    z2 = np.maximum(z2, 0)

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


def alternating_update_array(target_array, new_data_list, max_len=500):
    updated_array = np.concatenate((target_array, *new_data_list))

    if len(updated_array) > max_len:
        return updated_array[-max_len:]
    return updated_array


def find_center_of_mass(group):
    if not isinstance(group, np.ndarray):
        group = np.array(group)

    indices = np.arange(len(group))
    sum_of_values = np.sum(group)

    if sum_of_values == 0:
        return len(group) / 2

    weighted_sum = np.sum(indices * group)
    return weighted_sum / sum_of_values


# def array_modifier(data_arr, min_height=15, min_distance=10):
#
#     peak_positions = []
#     i = 0
#     while i < len(data_arr):
#         if data_arr[i] >= min_height:
#             group_start_index = i
#             while i < len(data_arr) and data_arr[i] > 0:
#                 i += 1
#             group_end_index = i
#             group = data_arr[group_start_index:group_end_index]
#             center_of_mass_rel = find_center_of_mass(group)
#             absolute_pos = group_start_index + center_of_mass_rel
#             peak_positions.append(absolute_pos)
#         else:
#             i += 1
#     return peak_positions

def array_modifier(data_arr, min_height=5, min_distance=80):
    peak_positions = []
    i = 0
    while i < len(data_arr):

        if data_arr[i] >= min_height:
            group_start_index = i

            if len(peak_positions) == 0 or (group_start_index - peak_positions[-1]) >= min_distance:
                peak_positions.append(float(group_start_index))

            while i < len(data_arr) and data_arr[i] > 0:
                i += 1
        else:
            i += 1

    return peak_positions


def co_determination(heart_peak_positions, ecmo_peak_positions):
    no_bpm = 7
    default_status = 0

    if len(heart_peak_positions) < 2:
        return no_bpm, default_status, False

    last_heart_peak = heart_peak_positions[-1]
    prev_heart_peak = heart_peak_positions[-2]

    relevant_ecmo_peak = None

    for peak in reversed(ecmo_peak_positions):
        if prev_heart_peak <= peak < last_heart_peak:
            relevant_ecmo_peak = peak
            break

    if relevant_ecmo_peak is None:
        return no_bpm, default_status, False

    heart_interval = last_heart_peak - prev_heart_peak
    if heart_interval <= 0:
        return no_bpm, default_status, False
    h_bpm = int(60 / (heart_interval * 0.004))

    if not (40 <= h_bpm <= 100):
        return no_bpm, default_status, False

    delay_from_prev_heart = relevant_ecmo_peak - prev_heart_peak
    delay_ratio = delay_from_prev_heart / heart_interval

    if 0 <= delay_ratio < 0.3:
        return h_bpm, 2, True  # lag
    elif 0.3 <= delay_ratio < 0.5:
        return h_bpm, 0, True  # stay
    elif 0.5 <= delay_ratio < 1.0:
        return h_bpm, 1, True  # lead
    else:
        return h_bpm, 0, True


def calculate_bpm_adjustment(heart_peak_positions, ecmo_peak_positions, previous_bpm):

    # 기본 반환값: 현재 BPM 유지, 계산 실패
    default_h_bpm = previous_bpm   # 실패 시 사용할 기본 BPM (혹은 이전 BPM 값)
    no_adjustment = 0

    if len(heart_peak_positions) < 2:
        return default_h_bpm, no_adjustment, False

    last_heart_peak = heart_peak_positions[-1]
    prev_heart_peak = heart_peak_positions[-2]

    # 두 심장 펄스 사이의 관련 ECMO 펄스 찾기
    relevant_ecmo_peak = None
    for peak in reversed(ecmo_peak_positions):
        if prev_heart_peak <= peak < last_heart_peak:
            relevant_ecmo_peak = peak
            break

    if relevant_ecmo_peak is None:
        return default_h_bpm, no_adjustment, False

    # 심장 박동 간격 및 BPM 계산
    heart_interval = last_heart_peak - prev_heart_peak
    if heart_interval <= 0:
        return default_h_bpm, no_adjustment, False

    h_bpm = int(60 / (heart_interval * 0.004))

    # 유효 BPM 범위 확인
    if not (40 <= h_bpm <= 100):
        return h_bpm, no_adjustment, False

    # ECMO 펄스의 상대적 위치(delay_ratio) 계산
    delay_from_prev_heart = relevant_ecmo_peak - prev_heart_peak
    delay_ratio = delay_from_prev_heart / heart_interval

    # 제안하신 단계별 BPM 조정 로직 적용
    adjustment = 0
    if 0 <= delay_ratio < 0.2:  # 동시박동: 강한 감속 필요
        adjustment = -3
    elif 0.2 <= delay_ratio < 0.3:  # 약한 동시박동: 중간 감속 필요
        adjustment = -1
    elif 0.3 <= delay_ratio < 0.5:  # 이상적인 역박동 구간: 유지
        adjustment = 0
    elif 0.5 <= delay_ratio < 0.55:  # 약한 동시박동: 중간 가속 필요
        adjustment = 1
    elif 0.55 <= delay_ratio < 0.8:  # 약한 동시박동: 중간 가속 필요
        adjustment = 2
    elif 0.8 <= delay_ratio <= 1.0:  # 동시박동: 강한 가속 필요
        adjustment = 3

    return h_bpm, adjustment, True