
peak_time = [f'{i}' for i in range(7, 21)]

def dataframe_summary(df):
    print(f"데이터 shape : {df.shape}")
    summary = pd.DataFrame(df.dtypes, columns=["Data_type"])
    summary = summary.reset_index()
    summary = summary.rename(columns={'index': 'feature'})
    summary['null_info'] = df.isnull().sum().values
    summary['unique_info'] = df.nunique().values
    summary['first_data'] = df.loc[0].values
    summary['second_data'] = df.loc[1].values
    summary['third_data'] = df.loc[2].values
    
    return summary

def date_transform(df):
    korea_holidays = holidays.KR(years = [2024])
    
    df['일시'] = pd.to_datetime(df['일시'], format = '%Y%m%d %H')

    # df['month'] = df['일시'].dt.month
    # df['day'] = df['일시'].dt.day
    # df['time'] = df['일시'].dt.hour


    df['date_only'] = df['일시'].dt.date # 시간을 제외한 날짜 형식만 저장
    df['is_holiday'] = df['date_only'].isin(korea_holidays).astype(int) # 공휴일 정보 매핑 
    
    df['month'] = (df['일시'].dt.month).astype(str).astype('category')
    df['day'] = (df['일시'].dt.day).astype(str).astype('category')
    df['time'] = (df['일시'].dt.hour).astype(str).astype('category')
    
    df['weekday'] = df['일시'].dt.weekday # 주중, 주말 구분 위해 생성
    
    df['is_offday'] = ((df['is_holiday'] == 1) | (df['weekday'] >= 5)).map({False : '0', True : '1'}).astype('category') # 주말이면서, 공휴일인 날 구분하기 위해 생성
    df['peak_time'] = ((df['is_holiday'] == 0) & (df['time'].isin(peak_time)) ).map({False : '0', True : '1'}).astype('category')
    
    df.drop(['일시', 'weekday', 'is_holiday','date_only', 'num_date_time'], axis =1, inplace=True)
    # df.drop('num_date_time', axis =1, inplace=True)
    
date_transform(train)
date_transform(valid)
date_transform(test)

##########################

# 불쾌 지수 파생변수 생성

train['불쾌지수'] = (train['기온(°C)'] - 0.55 * (1-0.01 * train['습도(%)']) * (train['기온(°C)'] - 14.5)).astype('int')
valid['불쾌지수'] = (valid['기온(°C)'] - 0.55 * (1-0.01 * valid['습도(%)']) * (valid['기온(°C)'] - 14.5)).astype('int')
test['불쾌지수'] = (test['기온(°C)'] - 0.55 * (1-0.01 * test['습도(%)']) * (test['기온(°C)'] - 14.5)).astype('int')

"""
불쾌지수(DI)
“Discomfort Index Calculator” –> DI = T − 0.55·(1 − 0.01·RH)·(T − 14.5)

DI < 21 -> 불쾌감 거의 없음,
21 =< DI < 24 -> 약간의 불쾌감,
24 =< DI < 26 -> 조금 더 강한 불쾌감,
26 =< DI -> 불쾌감 강함

"""


# 건물별 냉방면적 비율 생성
bd_info['cooling_ratio'] = bd_info['냉방면적(m2)'] / bd_info['연면적(m2)']


# 일사, 일조 주기 기반 생성 (case 03)
test['cycle_time'] = np.round(((np.sin(2 * np.pi * test['time'] / 24)) + 1) / 2, 4)
test['일조(hr)'] = 0 + (1-0)* test['cycle_time']
test['일사(MJ/m2)'] = 0 + (3.95-0)* test['cycle_time']


# 일사, 일조 건물별 특징 반영 생성 (case 04)
for building_num in train['건물번호'].unique():
    per_building_value = dict(train[train['건물번호'] == building_num].groupby('time')['일조(hr)'].mean())
    for i in range(len(test[test['건물번호'] == building_num])):
        test[test['건물번호'] == building_num][i] = per_building_value[test['time'][i]]

for building_num in train['건물번호'].unique():
    per_building_value = dict(train[train['건물번호'] == building_num].groupby('time')['일사(MJ/m2)'].mean())
    for i in range(len(test[test['건물번호'] == building_num])):
        test[test['건물번호'] == building_num][i] = per_building_value[test['time'][i]]
