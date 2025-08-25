# 건물별 전력소비량 예측 모델 개발 경진대회

- 2025년 8월에 열린 전력소비량 예측 경진대회 참가
- 사용한 모델 : Temporal Fusion Transformer(TFT)

- TFT 모델의 아키텍처 
<img width="1070" height="603" alt="image" src="https://github.com/user-attachments/assets/c722c670-76a3-4e56-b79c-281a83a2394c" />

- TFT 모델의 특징
  1. 변수 선택(Variable Selection) 네트워크 (VSN)
    - 각 시점 별 입력 데이터 중 어떤 피처가 중요한지 동적으로 자동 선택하는 것이 목적
    - 중요한 피처를 더욱 활용하여 모델 성능 높임
  2. 정적 변수 인코딩(Static Convariate Encoder)
    - 정적인 피처(건물 번호, 지역명 등) 정보를 임베딩하여 시계열 전체에 일관되게 반영하여, 건물 내 또는 지역 내와 같이 그룹별 고유한 패턴 반영 가능
  3. LSTM 기반 시계열 모델(인코더-디코더에 LSTM 사용)
    - 장기, 단기 패턴  포착
  4. 멀티-헤드 셀프 어텐션
    - 예측하고자 하는 값(미래 시점) 간에도 상호작용 가능
  5. Gated Residual 네트워크 (GRN)
    - 각 레이어에서 정보의 흐름을 조정하는 역할로, Redisual Connection을 통해 모델 안정성 보장
    - 효과적인 학습을 위한 구조이며 정보 손실 방지 
