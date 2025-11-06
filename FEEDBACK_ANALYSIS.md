# 문제 분석 및 개선 제안

## 문제 1: FNN RS 판정값이 낮아 심장 펄스 인식 실패

### 현재 문제점 분석

1. **RS 임계값 설정 문제**
   - 현재 `LOW_RS = 0.12`로 설정되어 있음
   - 실제 신호가 있어도 RS가 낮으면 배제됨
   - `est_h_ibp`, `est_e_ibp`가 보이는데도 BPM 계산이 안 되는 것은 RS 필터링 때문

2. **가상박 생성 로직의 한계**
   - 244-265라인: RS가 낮을 때 가상박을 생성하지만, 실제 `heart_peak_indices`에 추가되지 않음
   - 가상박을 `recent_events`에만 추가하고 `continue`로 BPM 계산을 건너뜀
   - 다음 사이클에서도 같은 문제가 반복될 수 있음

3. **연속 실패 대응 부족**
   - `hold_beat_count < 2`로 제한되어 연속 3회 이상 실패 시 더 이상 가상박 생성 안 함
   - 긴 공백 구간에서 대응이 어려움

### 개선 제안

#### 1-1. RS 임계값 동적 조정
```python
# 현재: 고정값
LOW_RS = 0.12

# 개선: 적응적 임계값
if self.last_valid_hh_s is not None:
    # 최근 유효 RS 값들의 평균을 기준으로 설정
    expected_rs = 0.15  # 기본값
    LOW_RS = max(0.08, expected_rs * 0.7)  # 더 낮은 임계값 허용
else:
    LOW_RS = 0.08  # 초기에는 더 낮은 임계값 사용
```

#### 1-2. RS 낮을 때도 조건부로 BPM 계산 허용
```python
# RS가 낮아도 est_h_ibp, est_e_ibp가 명확히 보이면 BPM 계산 진행
if rs_min < LOW_RS:
    # 추가 검증: peak 위치가 신호 파형과 일치하는지 확인
    if self.get_outp_h[h_prev_idx] > 0 and self.get_outp_h[h_curr_idx] > 0:
        # 신호는 있지만 RS만 낮은 경우: 신뢰도 낮춰서 사용
        confidence_factor = rs_min / LOW_RS  # 0~1 사이 값
        # 가상박 생성 대신, 낮은 신뢰도로 실제 이벤트 추가
        # 단, BPM 변화량 제한을 더 엄격하게
```

#### 1-3. 가상박을 실제 펄스로 활용
```python
# RS 낮을 때 가상박 생성 시, 다음 사이클에서 이를 활용
if rs_min < LOW_RS and self.last_valid_hh_s is not None:
    # 가상박 생성
    # ... 기존 로직 ...
    
    # 추가: 가상박을 heart_peak_indices에 추가 (다음 사이클용)
    virtual_h_prev = self.last_valid_h_curr_idx
    virtual_h_curr = virtual_h_prev + round(self.last_valid_hh_s / self.DT)
    # 다음 사이클에서 이 값을 활용할 수 있도록 저장
```

#### 1-4. array_modifier의 min_height 조정
```python
# Function.py의 array_modifier 호출 시
# 현재: min_height=7 (기본값)
# 개선: RS 낮을 때를 대비해 더 낮은 임계값도 시도
h_idx_raw = array_modifier(self.get_outp_h, min_height=7)
if len(h_idx_raw) < 2:
    # min_height를 낮춰서 재시도
    h_idx_raw = array_modifier(self.get_outp_h, min_height=5)
```

---

## 문제 2: DNN 5개 펄스 제어 문제

### 현재 문제점 분석

1. **초기 5개 이벤트 채우기 전 제어 불가**
   - 142-143라인: `SEQ_LENGTH = 5`, 초기값으로 `(0.35, 0.65, 1.0, 1.0, 0.5, 0.5, 0.5)` 채움
   - 하지만 실제 펄스가 2개만 있어도 바로 DNN에 전달 (268-271라인)
   - 초기 5개가 실제 이벤트로 채워지기 전까지는 제대로 된 제어가 어려움

2. **가상박과 실제 펄스의 불일치**
   - 가상박을 만들어도 DNN의 입력이 실제 상황과 맞지 않을 수 있음
   - 가상박의 RS 값이 낮아서 (0.5 배율 적용) DNN이 이를 신뢰하지 않을 수 있음

3. **DNN 신뢰도와 강제 제어의 충돌**
   - 291라인: `conf < 0.45`일 때 강제 제어 사용
   - 하지만 5개 이벤트가 채워지지 않았을 때는 conf가 낮을 수밖에 없음

### 개선 제안

#### 2-1. 초기 5개 이벤트 채울 때까지 보수적 제어
```python
# recent_events가 실제 이벤트로 채워졌는지 확인
actual_event_count = len([e for e in recent_events 
                          if e[4] > 0.1 and e[5] > 0.1 and e[6] > 0.1])  # RS 값 체크

if actual_event_count < 5:
    # 아직 5개가 안 채워졌으면: 보수적 제어 (현재 BPM 유지 또는 작은 조정만)
    # DNN 출력을 사용하되, 더 강한 스무딩 적용
    if conf < 0.6:  # 더 높은 임계값
        # 현재 ratio만 보고 간단한 제어
        if ratio_display < 0.30:
            current_action_index = 3  # +1 가속
        elif ratio_display > 0.40:
            current_action_index = 1  # -1 감속
        else:
            current_action_index = 2  # 유지
```

#### 2-2. 가상박의 신뢰도 향상
```python
# 현재: 가상박 생성 시 RS 값을 0.5 배율로 낮춤
pad_event = (phi_v, 1.0 - phi_v, 1.0, 1.0,
             rs_h_prev * 0.5, rs_e_val * 0.5, rs_h_curr * 0.5)

# 개선: 가상박이라도 이전 유효 이벤트의 RS 평균 사용
if self.last_valid_rs_avg is not None:
    avg_rs = self.last_valid_rs_avg
else:
    avg_rs = 0.15  # 기본값
pad_event = (phi_v, 1.0 - phi_v, 1.0, 1.0,
             avg_rs, avg_rs, avg_rs)  # 더 현실적인 RS 값
```

#### 2-3. 이벤트 버퍼 관리 개선
```python
# 초기값 대신, 실제 이벤트가 들어올 때까지 대기
if not self.ai_started:
    # 첫 5개 실제 이벤트를 기다림
    if len(recent_events) >= 5 and all(e[4] > 0.1 for e in list(recent_events)[-5:]):
        self.ai_started = True
    else:
        # 아직 준비 안 됨: 보수적 제어
        send_bpm.put(self.bpm)
        send_status.put(0)
        continue
```

#### 2-4. 가상박과 실제 펄스의 혼합 전략
```python
# 가상박을 만들되, 실제 펄스가 감지되면 즉시 대체
# 가상박은 "임시 보간"으로만 사용하고, 실제 펄스가 나오면 recent_events 업데이트

# recent_events에 가상박 표시 플래그 추가
virtual_event = (phi_v, 1.0 - phi_v, 1.0, 1.0, avg_rs, avg_rs, avg_rs, True)  # 마지막: 가상 여부
# 실제 이벤트가 들어오면 가상박을 대체
```

---

## 통합 개선 전략

### 전략 1: RS 임계값 적응적 조정
- 최근 유효 RS 값들의 통계를 추적
- 동적으로 임계값 조정
- 신호 품질이 낮을 때는 더 관대한 임계값 사용

### 전략 2: 다단계 펄스 감지
1. 1차: 높은 RS로 엄격하게 감지
2. 2차: RS 낮아도 신호 파형 확인
3. 3차: 가상박 생성 (마지막 수단)

### 전략 3: DNN 초기화 개선
- 실제 이벤트 5개가 채워질 때까지 보수적 제어
- 가상박은 "플레이스홀더"로만 사용
- 실제 이벤트가 들어오면 즉시 업데이트

### 전략 4: BPM 연속성 보장
- RS 낮아도 이전 BPM 값을 유지하되, 작은 조정은 허용
- 긴 공백 구간에서도 이전 패턴 유지

---

## 구현 우선순위

1. **높은 우선순위** (즉시 효과)
   - RS 임계값 동적 조정 (1-1)
   - 초기 5개 이벤트 채울 때까지 보수적 제어 (2-1)
   - array_modifier min_height 조정 (1-4)

2. **중간 우선순위** (안정성 향상)
   - 가상박 신뢰도 향상 (2-2)
   - 이벤트 버퍼 관리 개선 (2-3)

3. **낮은 우선순위** (장기 개선)
   - 가상박을 실제 펄스로 활용 (1-3)
   - 다단계 펄스 감지 (전략 2)

