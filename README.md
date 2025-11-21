# 건물별 전력소비량 예측 모델링

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

6. 다중 시점 예측( Multi-Horizon Forecast)에 좋은 성능을 보임
    - 모델링의 목적이 8월 25일~31일 사이 1시간 단위 전력 소비량을 예측한다는 점에서 Temporal Fusion Transformer 선택


##
<br>
<br>


(데이콘 경진대회 데이터를 활용하였습니다.)

###  건물 별, 1시간 단위 전력 소비량 예측 모델 개발

- DATA
-> 제공 피처 수 → 10개 (온도, 습도, 건물 번호, 날짜, 강수량, 일조, 일사량 등)
-> 204,000개(Train), 16,800개(Test)

</br>

- 특이사항
-> Train Data에는 일사, 일조 데이터가 있으나 Test Data에는 없음
-> 대회 측에서 테스트 데이터의 실제 target 값은 제공하지 않아, 훈련 데이터를 훈련-검증-테스트로 분할하여 모델링 성능 점검
-> 기존 데이터 성질을 유지하기 위해 테스트로 분할된 데이터는 일조, 일사량을 전부 제거한 상태로 만듦

</br>

- 건물 별 정보
→ 연면적, 냉방면적, 태양광 용량, PCS 용량, ESS 용량 (5개 피처)


※ 기본 변수만 사용했을 때,  주어진 데이터 기반 파생 변수 생성했을 때 성능 비교 ※

++ 공정하게 비교하기 위해 모델 파라미터는 동일하게!

---

#### 비교 모델 간 공통 과정

- 고유 값 , 결측 값 여부, 데이터  형태 확인

```python
def DataFrame_Summary(df):
	print(f'데이터 Shape: {df.shape}')
	summary= pd.DataFrame(df.dtypes, columns = ['data_type'])
	summary.reset_index(inplace = True)
	summary = summary.rename(columns = {'index': 'feature'})
	summary['NA_info'] = summary.isnull().sum().values
	summary['unique_info'] = summary.nunique().values
	summary['1st_row'] = summary.loc[0].values
	summary['2nd_row'] = summary.loc[1].values
	summary['3rd_row'] = summary.loc[2].values
	
	return summary
```

- 시계열 데이터 처리를 위한 time_idx 생성
-> 건물 별 time_idx가 연속하도록 인덱스 부여

- TimeSeriesDatasets
-> 훈련 데이터를 기반으로 Datasets을 구성
-> 검증, 테스트 데이터는 훈련데이터셋.from_datasets()을 사용하여 생성
> - 훈련 데이터를 훈련 + 검증으로 분할하기 전, TimeSeriesDataset은 예측하기 시점의 이전 시점을 encoder_length로 사용  
> - 특히 테스트 데이터는 전체를 전부 예측하기 때문에 encoder_length가 0이 되는 문제가 발생하고, 이를 해결하기 위해 훈련 데이터를 분할하기 전에 일부를 떼어 테스트 데이터의 encoder_length로 사용하였다.


- 사용 모델
-> 트랜스포머 기반 시계열 모델인 Temporal Fusion Transformer(TFT) 사용
-> TFT는 Lim(2021)이 제안한 모델로, 특히 다중 시점을 예측에 좋음
-> 모델 내부에 LSTM 모듈을 사용하여 장기-단기 패턴 동시에 포착하는 장점
-> 사용한 TFT 파라미터 (By GridSearch)
```python
      select_model_params = {
      'learning_rate': 4e-3,
      'dropout': 0,
      'lstm_size': 80,
      'multi_head_attention': 1,
      'continuous_size': 32
      }
      epochs = 50
      patience = 5
      batch_size = 128
   ```
    
</br>

---

</br>


### CASE 01. 기본 제공된 피처만 사용(일사, 일조 피처 x)

- 기존 데이터에서 일사, 일조 컬럼만 제거한 성능
- 제공된 데이터 그대로 사용

Validation Data SMAPE : `0.116`
Test Data SMAPE : `0.129`

</br>

---

</br>


### CASE 02. 기본 변수 + 파생 변수포함(일사, 일조 제거)

- 생성할 변수 
-> 불쾌지수를 나타내는 `Discomfort Index(DI)`
-> 휴일을 구분하는 `is_offday`
-> 전력소비량이 많은 시간을 구분하는 `peak_time`




`1. Discomfort Index`
-> 6~8월은 여름 절기라 불쾌지수도 전력 소비량에 영향을 끼칠 것으로 예상
```
DI < 21 -> 불쾌감 거의 없음
21 <= DI < 24 -> 약간의 불쾌감
24 <= DI < 26 -> 보다 강한 불쾌감
26 <= DI 강한 불쾌감
```
|참고 자료: https://www.donga.com/news/Economy/article/all/20000621/7549296/1


</br>

`2. is_offday`
-> 토요일, 월요일 또는 특정 공휴일은 전력소비량이 평소보다 적을 것으로 예상, 휴일을 구분하는 변수
```python
korea_holidays = holidays.KR(years = [2024])
df['is_holiday'] = df['data_only'].isin(koear_holidays).astype(int)
df['is_offday'] =  ((df['is_holiday'] == 1) | (df['weekday'] >= 5)).map({False: '0', True : '1'}).astype('category')
```

</br>

`3. peak_time`
-> 시간대 별 소비량 구분

![](https://velog.velcdn.com/images/yjhut/post/c1fe1fbf-f3bb-4c9c-9058-19094921050f/image.png) 

위 그래프는 시간 기준 평균 소비량 시각화 결과.... 7~20시까지 peak_time : 1 그외 시간은 0


Validation Data SMAPE : `0.075`
Test Data SMAPE : `0.092`

</br>

---

</br>


### CASE 03. 기본 변수 + 파생 변수포함 + 일사, 일조 생성

- CASE 02에서 생성한 파생변수에 테스트 데이터에도 훈련 데이터 기반의 일사, 일조 컬럼 생성 

- 일사, 일조 시간에 따라 사인 함수로 생성
-> 훈련 데이터 기반으로 각 일사, 일조 최솟값, 최댓값 추출
$$
    value = min + (max - min) * cycle_{time}
    $$   
-> 보통 일출 전, 일몰 후에는 없기 때문에 여름 절기 고려 + case 02에서 생성한 peak_time이외의 시간은 일조, 일사 둘 다 0이 되도록 후처리
-> 단순 주기에 의존한 일사, 일조량을 생성했을 때의 결과

</br>


Validation SMAPE : `0.069`
Test SMAPE : `0.093`

</br>

---

</br>


### CASE 04. 기본 변수 + 파생 변수포함 + 일사, 일조 생성 추가 방법 + 스케일러 적용

- CASE 03의 방법은 단순 주기함수로 값을 부여한 경우인데, 검증 성능은 개선되었으나 결과적으로 테스트 셋에서 성능 개선이 없었다.
- 그렇다면 일조, 일사량 생성 방법을 변경해보자
>🔎
> - 현재 데이터가 건물별로 구분되어 있음. 건물의 위치에 따라 당연히 일사, 일조량이 다르므로 당연히 건물별 특징을 고려해야함
> - 아래는 훈련 데이터에서 15개 건물에 대해 일사, 일조량을 시간대별 시각화 한 결과이다.
>![](https://velog.velcdn.com/images/yjhut/post/18b37043-8a3f-4c14-801f-f78605ab1c41/image.png)|시간별 일조량
>
>![](https://velog.velcdn.com/images/yjhut/post/ff272b27-49dd-4100-b059-ca1cacdfb53e/image.png)|시간별 일사량


- 위 시각화 결과에 따라, 건물 별, 시간 별 일조, 일사량의 평균을 구하여 건물, 시간에 맞게 생성

```python
for building_num in train['건물번호'].unique():
	per_building_value = dict(train[train['건물번호'] == build_ing_num].groupby('time')['일조(hr')].mean())
    for i in range(len(test[test['건물번호'] == building_num])):
    test[test['건물번호'] == building_num][i] = per_building_value[test['time'][i]]
    
# 일사량도 동일하게 생성 
```

(++ 스케일 맞추기 위해 단위가 큰 변수는 MinMax Scaler, Standard Scaler, Robust Scaler를 적용)

</br>

스케일러 적용 x 

Validation SMAPE : `0.068`
Test SMAPE : `0.085`



스케일러 적용 o

Validation SMAPE : `0.065`
Test SMAPE : `0.082`


</br>

---




#### 최종 성능 정리

|-|CASE_01|CASE_02|CASE_03|CASE_04|
|:---:|:---:|:---:|:---:|:---:|
|Validation|0.116|0.075|0.069|<span style= "color:yellow">0.065</span>|
|TEST|0.129|0.092|0.093|<span style= "color:yellow">0.082</span>|

`평가지표: SMAPE`

</br>


- 개별 프로젝트를 통해 얻게된 점

> - 휴일 여부, 피크 타임 등 데이터와 목표의 특성을 고려한 파생변수 생성은 적용하지 않았을 때보다 큰 성능 향상을 보여, 데이터 EDA가 적절한 모델을 고르는 것만큼 중요성을 가진다는 것
 
> - 시계열 데이터를 처리하기 위해 TimeSeriesDataset을 구성할 때 검증, 테스트 데이터는 전체를 예측 해야하기 때문에 min_encoder_length를 미리 생성하여 모델이 동작할 수 있도록 조치

> - 일사, 일조와 같이 주기성을 갖는 피처는 프로젝트에서 사용한대로 주기 함수를 이용한 방법과 데이터에서 주어진 특징을 활용한 생성 방법이 있고, 이번에는 후자의 성능이 좋았으나 전자의 방법을 활용한 개선 아이디어도 좋은 결과를 보일 수 있을 것 같다.



