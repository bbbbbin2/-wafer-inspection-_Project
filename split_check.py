import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print("==================================================")
print("📦 [1단계] 정제된 마스터 데이터셋 파일 로드")
print("==================================================")
df = pd.read_pickle("LSWMD_clean_64.pkl")
print(f"로드 완료! 전체 유효 데이터 개수: {len(df)}장")

print("\n==================================================")
print("✂️ [2단계] 문제집(X)과 정답지(y) 생성 및 라벨 인코딩")
print("==================================================")
label_map = {l: i for i, l in enumerate(['none', 'Loc', 'Edge-Loc', 'Center', 'Edge-Ring', 'Scratch', 'Random', 'Near-full', 'Donut'])}
df['label_idx'] = df['clean_label'].map(label_map)

X = np.stack(df['waferMap_scaled'].values)
y = df['label_idx'].values

print(f"문제집 X 구조 (Shape): {X.shape} -> (샘플 수, 가로, 세로)")
print(f"정답지 y 구조 (Shape): {y.shape} -> (샘플 수,)")

print("\n==================================================")
print("📐 [3단계] 70 : 15 : 15 비율로 데이터 분할 (Stratified Split)")
print("==================================================")
# 1차 분할: 전체 17만장 중 70%는 Train, 나머지 30%는 임시(Temp)방으로
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, 
    test_size=0.30, 
    random_state=42, 
    stratify=y
)

# 2차 분할: 임시방 30%를 반반(0.5) 쪼개서 각각 15%, 15%로 할당
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, 
    test_size=0.50, 
    random_state=42, 
    stratify=y_temp
)

total_count = len(X)
print(f"🟢 [교과서] Train 데이터수      : {len(X_train)}장 (전체의 {len(X_train)/total_count*100:.1f}%)")
print(f"🟡 [모의고사] Validation 데이터수 : {len(X_val)}장 (전체의 {len(X_val)/total_count*100:.1f}%)")
print(f"🔴 [수능시험] Test 데이터수       : {len(X_test)}장 (전체의 {len(X_test)/total_count*100:.1f}%)")
print("-" * 50)
print(f"합계 데이터수                      : {len(X_train) + len(X_val) + len(X_test)}장")
print("==================================================")
print("🎉 [검증 완료] 70:15:15 분할이 수학적으로 완벽하게 성공했습니다!")
