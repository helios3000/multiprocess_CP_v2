# 데이터 저장 구조 분석 및 수정 시 주의사항

## 현재 데이터 저장 구조

### 1. 저장 메커니즘
```
determination() 함수 (Thread 1)
  ↓
  decision_results_queue.put()  ← 펄스 감지 시마다 호출
  ↓
save_file() 함수 (Thread 2)
  ↓
  매 4개 샘플마다 decision_results_queue.get_nowait() 시도
  ↓
  있으면: ['', '', '', raw_bpm] 형태로 저장
  없으면: [''] * 4 형태로 저장 (빈 값)
```

### 2. 현재 `decision_results_queue.put()` 호출 위치

**경로 1: 가상박 생성 시 (258라인)**
```python
if rs_min < LOW_RS and ...:
    decision_results_queue.put({...})  # ✅ 호출
    continue  # BPM 계산 건너뜀
```

**경로 2: 실제 이벤트 처리 시 (320라인)**
```python
decision_results_queue.put({...})  # ✅ 호출
send_bpm.put(self.bpm)
send_status.put(bpm_adjustment)
```

**경로 3: BPM 계산 실패 시 (194, 200, 207, 330, 334라인)**
```python
send_bpm.put(self.bpm)
send_status.put(0)
continue  # ❌ decision_results_queue.put() 호출 안 함!
```

### 3. 잠재적 문제점

#### 문제 1: `decision_results_queue.put()` 호출 불일치
- 펄스 감지 실패 시 (경로 3): `put()` 호출 안 함
- 하지만 `save_file()`은 계속 `get_nowait()` 시도
- 결과: 빈 값이 저장됨 (정상 동작이지만, 데이터와 결정이 불일치)

#### 문제 2: 수정 시 주의사항
- 새로운 `continue` 경로 추가 시: `decision_results_queue.put()` 호출 여부 결정 필요
- `put()` 호출 횟수 증가 → 큐에 쌓일 수 있음 (큐 크기 제한 없음, 메모리 문제 가능)
- `put()` 호출 횟수 감소 → 빈 값 증가 (데이터 분석 시 주의)

## 수정 시 안전한 방법

### ✅ 안전한 수정 (데이터 저장 구조 유지)

#### 1. RS 임계값 조정만 하는 경우
```python
# 현재
LOW_RS = 0.12

# 수정
LOW_RS = 0.08  # 또는 동적 값
```
**영향**: `decision_results_queue.put()` 호출 횟수는 변하지 않음 → **안전**

#### 2. array_modifier 이중 시도
```python
h_idx_raw = array_modifier(self.get_outp_h, min_height=7)
if len(h_idx_raw) < 2:
    h_idx_raw = array_modifier(self.get_outp_h, min_height=5)
```
**영향**: 펄스 감지 성공률 증가 → `put()` 호출 증가 가능, but 큐 처리로 문제 없음 → **안전**

#### 3. RS 낮아도 조건부 BPM 계산 허용
```python
# 현재: RS 낮으면 가상박 생성 후 continue
if rs_min < LOW_RS:
    # 가상박 생성
    decision_results_queue.put({...})  # ✅ 호출
    continue

# 수정: RS 낮아도 신호 확인 후 조건부 처리
if rs_min < LOW_RS:
    if 신호_파형_확인():  # 추가 검증
        # 가상박 대신 실제 이벤트로 처리 (신뢰도 낮춰서)
        decision_results_queue.put({...})  # ✅ 호출 유지
        # BPM 계산 계속 진행
    else:
        # 가상박 생성
        decision_results_queue.put({...})  # ✅ 호출 유지
        continue
```
**영향**: `put()` 호출 횟수는 유지, 단지 처리 방식만 변경 → **안전**

### ⚠️ 주의가 필요한 수정

#### 1. 새로운 continue 경로 추가 시
```python
# ❌ 나쁜 예: put() 호출 없이 continue
if some_condition:
    send_bpm.put(self.bpm)
    send_status.put(0)
    continue  # decision_results_queue.put() 없음!

# ✅ 좋은 예: put() 호출 후 continue
if some_condition:
    decision_results_queue.put({
        'raw_bpm': self.bpm, 'adj': 0,
        'he': 0, 'eh': 0,
        'rs_hp': 0, 'rs_e': 0, 'rs_hc': 0,
        'ai_start': ''
    })
    send_bpm.put(self.bpm)
    send_status.put(0)
    continue
```

#### 2. DNN 초기화 로직 추가 시
```python
# 초기 5개 이벤트 채울 때까지 보수적 제어
if not self.ai_started:
    if len(recent_events) < 5:
        # ❌ 나쁜 예: put() 없이 continue
        send_bpm.put(self.bpm)
        send_status.put(0)
        continue
        
        # ✅ 좋은 예: put() 호출 후 continue
        decision_results_queue.put({
            'raw_bpm': self.bpm, 'adj': 0,
            'he': 0, 'eh': 0,
            'rs_hp': 0, 'rs_e': 0, 'rs_hc': 0,
            'ai_start': 'INIT'
        })
        send_bpm.put(self.bpm)
        send_status.put(0)
        continue
```

## 권장 수정 전략

### Phase 1: 안전한 수정 (즉시 적용 가능)
1. ✅ RS 임계값 조정: `LOW_RS = 0.08` 또는 `0.10`
2. ✅ array_modifier 이중 시도
3. ✅ 주석 추가로 코드 가독성 향상

### Phase 2: 조건부 수정 (테스트 필요)
1. ⚠️ RS 낮아도 조건부 BPM 계산 허용
2. ⚠️ DNN 초기화 로직 개선
3. ⚠️ 가상박 신뢰도 향상

### Phase 3: 구조적 개선 (장기)
1. 🔄 모든 `continue` 경로에서 `decision_results_queue.put()` 호출 보장
2. 🔄 큐 크기 모니터링 추가
3. 🔄 저장 데이터와 결정 로그의 일치성 검증

## 체크리스트

수정 전 확인:
- [ ] 새로운 `continue` 경로 추가 시 `decision_results_queue.put()` 호출하는가?
- [ ] `put()` 호출 시 필수 필드가 모두 포함되는가?
- [ ] `save_file()`의 `get_nowait()` 예외 처리로 인해 크래시는 없는가?

수정 후 확인:
- [ ] 프로그램이 정상 종료되는가?
- [ ] CSV 파일이 정상적으로 저장되는가?
- [ ] CSV 파일의 열 개수가 일치하는가? (19개: spo2 ~ ai_start)
- [ ] `decision_results_queue`에 데이터가 과도하게 쌓이지 않는가?

