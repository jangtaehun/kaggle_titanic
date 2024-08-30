### 👨‍🏫 Titanic - Machine Learning from Disaster
kaggle에서 제공하는 Tatanic data를 이용해 EDA와 model 학습을 통해 생존자를 예측하는 프로젝트

---
### ⏲️ 분석 기간
2024.08.20 - 2024.08.21

---

### 📝 소개
타이타닉은 세계에서 가장 유명한 침몰선이라 할 수 있으며, 사고로부터 100년이 넘게 지난 오늘날까지도 관련 연구가 활발하게 이루어지며 대중매체 등에서 많이 다뤄지고 있습니다. 그래서 많은 사람들이 머신러닝 학습을 처음 시작할 때 Kaggle에서 제공하는 타이타닉 데이터를 이용해 Kaggle 자체 대회에 참여하고 있습니다.
따라서 필자 역시 타이타닉 데이터를 이용한 Kaggle 타이타닉 대회에 참여해 가능한 높은 점수를 얻어보고자 이 프로젝트를 시작하게 되었습니다.

---

### 프로젝트 개요
##### 📌 목표
이 프로젝트의 목표는 타이타닉 탑승객의 생존 여부를 다양한 특징들을 바탕으로 예측하는 것입니다. 데이터셋은 탑승객의 인구통계, 티켓 등급, 선실 정보 등을 포함하고 있습니다. 이러한 특징들을 분석함으로써, 탑승객의 생존 여부를 정확하게 예측하는 모델을 개발하고자 합니다.

##### 🖥️ 데이터셋 (Data Set)
이 프로젝트에서 사용한 데이터셋은 Kaggle에서 제공하는 다음 파일들로 구성되어 있습니다.
1. titanic_train.csv: 훈련 데이터셋, 특징들과 목표 변수(Survived)를 포함.
2. test.csv: 테스트 데이터셋, 예측을 위해 사용될 데이터.
3. gender_submission.csv: 예측 결과를 제출하기 위한 샘플 파일.

---

##### 방법론
1. titanic에 대한 정보 수집
  * 문제 정의
  * 분석 대상에 대한 이해
2. titanic data set을 이용한 EDA
  * 공통 코드
  * titanic data에 대한 기본적인  정보
  * 통계 및 시각화
    * 여성과 아이들
    * 나이
    * 사회적 지위
    * Embarked(중간 정착 항구)
    * Cabin(선실 번호)
    * SibSp, Parch(같이 탑승한 형제자매 또는 배우자 인원수, 같이 탑승한 부모님 또는 어린이 인원수)
3. 모델 학습
  * RandomForest
  * XGBoost
  * LightGBM
  * CatBoost
4. 결론
  * 한계점

---

##### titanic에 대한 정보 수집
   ###### 문제 정의
titanic data set을 이용한 Kaggle에서 진행하는 대회는 생존자를 예측하는 문제이다.
* PassengerId: 탑승자 데이터 일련번호
* Survived: 생존 여부, 0 = 사망, 1 = 생존
* Pclass: 티켓의 선실 등급, 1 = 일등석, 2 = 이등석, 3 = 삼등석
* Sex: 탑승자 성별
* Name: 탑승자 이름
* Age: 탑승자 나이
* SibSp: 같이 탑승한 형제자매 또는 배우자 인원수
* Parch: 같이 탑승한 부모님 또는 어린이 인원수
* Ticket: 티켓 번호
* Fare: 요금
* Cabin: 선실 번호
* Embarked: 중간 정착 항구, C = Cherbourg, Q = Queenstown, S = Southampton
titanic_train.csv을 기반으로 titanic에 대한 지식과 적절한 EDA를 진행한 후 test.csv의 데이터를 이용해 예측한 후 결과를 gender_submission.csv와 결합한 후 제출하고 제출하는 문제이다.
   ###### 분석 대상에 대한 이해
titanic data set을 이용한 Kaggle에서 진행하는 대회는 생존자를 예측하는 문제이다.
* 길이: 269.1m
* 폭: 28m
* 높이: 53.3m
* 배수량: 52,310t
* 총 톤수: 46,328 GRT
* 최대 속도: 23노트 (43㎞/h)
* 최대 탑승 가능 인원 = 3,547명(승선객, 승무원 모두 포함)
* 선실 수
  * 1st-class(1등실): 416개
  * 2nd-class(2등실): 162개
  * 3rd-class(3등실): 269개
  * cabin area(전용실, 갑판실 등): 40개
* 층별 구조
   * 보트 갑판: 최상층으로 구명보트가 배치되어 있다.
   * 산책로가 있으며, 1등실, 2등실, 상선사관 등 산책로의 영역이 정해져 있다.
   * 1등실 산책로는 구명정이 비치되어 있지 않다.
   * A갑판: 산책 갑판
   * 거의 모든 영역이 1등실 전용이었다.
  * B갑판: 선교루 갑판
    * 객실은 모두 1등실이였으며 2개의 특별 객실들은 전용 테라스 및 산책로를 보유했다.
  * C갑판
    * 선두 - 선원들의 숙소
    * 선미 - 3등실 전용 휴게실
  * D갑판: 공공시설
    * 1등실 대합실
    * 2등실 식당, 2등실 식당
    * 3등실을 위한 공간도 마련되어 있어 연회 장소로 사용
  * E갑판
    * 1, 2, 3등실 모두의 객실들과 선원들의 숙소
  * F갑판
    * 객실은 3등실이 대부분이며 2등실, 선원들의 숙소도 있었다.
  * G갑판
    * 수면 위에서 가장 낮은 층으로 선원, 3등실 승객들의 객실이 있는 가장 낮은 갑판
  * 최하 갑판
    * 창고가 위치한 장소
    * 탱크 톱
    * 보일러실과 기관실이 위치한 장소
* 승객
  * 총 1,317명
  * 1등실 - 329명
    * 부유한 승객들이 주로 타고 있었다.
    * 객실 - 보트 갑판(최상층) ~ E갑판(상갑판)
  * 2등실 - 285명
    * 중산층 승객들이 주로 타고 있었다.
    * 객실 - D ~ F
  * 3등실 - 710명
    * 가난한 승객들이 주로 타고 있었다.
    * 당시 기준으로 하층민들이 주로 사용해 건강상태가 좋지 않았으며 이민자들이 많았던 만큼 배에 탑승하기 전에는 검역 과정을 걸쳤다.
    * 여자와 남자는 배의 앞머리와 뒷머리에 각각 따로 떨어져 승선했으나 가족 단위일 경우 같이 승선할 수 있었다.
* 요금
   * 1등석: 30파운드(150달러), 스위트 1등석은 870파운드(4350달러)
   * 2등석: 12파운드(60달러)
   * 3등석: 7파운드(35달러)
* 중간 정착 항구 및 최종 정착 항구
   * S = 영국 Southampton
   * C = 프랑스 Cherbourg
   * Q = 아일랜드 Queenstown
   * 최종 정착 항구 = 미국 New York
 
   *  ***승무원의 경우 갑판부, 기관부, 사주부가 있으나 Kaggle에서 제공하는 data set에서는 승무원들에 대한 정보가 없기 때문에 생략하겠습니다.***

* 충돌 및 탈출
   * 우현측면이 빙산과 충돌
   * 선원들이 여자와 아이들을 먼저 태울 것을 건의했으며, 선장은 승인했다. 하지만 소통의 오류로 ‘여성과 아이들만’으로 전달되어 여성과 어린이만 태웠기 때문에 자리가 있었음에도 남자는 승무원들이 거부해 구명보트 정원의 절반도 못 태운채 보트가 있었다. 뿐만 아니라 1,178명 정도를 태울 수 있는 구명보트만 구비되어 있었기 때문에 큰 인명 피해가 발생했다.

---

##### titanic data set을 이용한 EDA
titanic에 대한 정보를 바탕으로 여성과 아이들의 구조율이 높다는 것을 예측할 수 있다. 따라서 ‘여성과 아이들’에 집중을 해서 통계 및 시각화를 진행보고자 한다. 이를 위해 나이와 성별에 해당하는 feature를 사용할 것이다. 뿐만 아니라 ‘여성과 아이들’에만 초점을 맞추는 것이 아닌 사회적 지위에 따른 구조율도 확인해 볼 예정이다. 그 이유는 다음과 같다.
1. 선실 등급이 높을 수록 배 위쪽에 위치해 있다. 즉, 빙산이 충돌한 배 하층 부분에 비해 상층 부분은 대피할 수 있는 시간이 충분했다고 판단하고 있기 때문이다.
2. 선실 등급이 높다는 것은 당시 사회적 지위가 상당히 높다는 것이다. 즉, 그만큼의 대우를 받았다고 생각하고 있기 때문이다.
위의 두 가지 이유로 사회적 지위에 따른 구조율도 확인해 보고자 한다. 이를 위해 선실 등급과, 요금에 해당하는 feature를 사용할 것이다.

---

   ###### 공통 코드
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv('./titanic/titanic_train.csv')
predict_df = pd.read_csv('./titanic/test.csv')
gender_submission_df = pd.read_csv('./titanic/gender_submission.csv')
```
kaggle에서 제공하는 titanic data를 불러들이는데 사용하는 코드로 앞으로 사용되는 data와 해당 data를 각각 train_df, predict_df, gender_submission으로 선언한 부분이다.

   ###### titanic data set에 대한 기본적인 정보
```
print("train_df 데이터의 행 개수:", len(train_df))
print('train_df: 데이터 세트 Null 값 갯수 ',train_df.isnull().sum().sum())
print(train_df.isnull().sum())
print(train_df.columns)
print("------------------------------------------------------------ \n\n")

print("predict_df 데이터의 행 개수:", len(predict_df))
print('predict_df: 데이터 세트 Null 값 갯수 ',predict_df.isnull().sum().sum())
print(predict_df.isnull().sum())
print(predict_df.columns)
print("------------------------------------------------------------ \n\n")
```
![image](https://github.com/user-attachments/assets/58009547-1ccf-4088-966c-4cf510d0a057)

다음과 같은 정보를 확인할 수 있다. train_df에는 총 12개의 feature가 있으며, Age에 177개, Cabin에 687개, Embarked에 2개의 NaN 값이 있다는 것을 알 수 있다. test_df에도 NaN 값이 있지만, train_df와 같은 feature에 있는 것을 통해 train_df에서 NaN 값을 제거하고자 한 방법을 그대로 적용하면 될 것이라고 판단된다.
또한, 찾아 볼 수 있는 점으로 타이타닉에는 선원들을 제외한 총 1,317명이 탑승했지만 train_df, test_df를 합쳤을 때 총 1,309명으로 8명이 없다는 것을 알 수 있다. 이 부분에 대해서는 titanic data를 제공한 kaggle만이 이유를 알 것이다.

   ###### 통계 및 시각화
1. 여성과 아이들
여성의 구조율을 확인하기 먼저 확인해야 할 것은 여성과 남성의 수를 확인해 보는 것이다.
```
train_df['Sex'].value_counts()
```
남성은 577명, 여성은 314명으로 총 891명인 것을 확인할 수 있다. 좀 더 구체적으로 확인해 보겠다.
```
print(train_df.groupby(['Sex','Survived'])['Survived'].count())
print("\n-------------------------------------------------------------\n")

female = train_df[train_df['Sex'] == 'female'].shape[0]
female_0 = train_df[(train_df['Sex'] == 'female') & (train_df['Survived'] == 0)].shape[0]
female_1 = train_df[(train_df['Sex'] == 'female') & (train_df['Survived'] == 1)].shape[0]

male = train_df[train_df['Sex'] == 'male'].shape[0]
male_0 = train_df[(train_df['Sex'] == 'male') & (train_df['Survived'] == 0)].shape[0]
male_1 = train_df[(train_df['Sex'] == 'male') & (train_df['Survived'] == 1)].shape[0]

print(f"여성 생존률: {round(female_1 / (female_0 + female_1) * 100, 2)}")
print(f"남성 생존률: {round(male_1 / (male_0 + male_1) * 100, 2)}")
```
여성은 81명이 사망, 233명이 생존했다. 반면 남성은 468명이 사망, 109명이 생존한 것을 확인할 수 있다. 이렇게 나타난 수치를 비율을 통해서 남여 구조율을 비교하면 여성 생존률: 74.2 / 남성 생존률: 18.89로 여성의 구조율이 남성의 구조율보다 월등히 높다는 것을 알 수 있다. 시각화를 하면 다음과 같다.
```
custom_palette = ["#FFA07A", "#AFEEEE"]
sns.barplot(x='Sex', y = 'Survived', data=train_df, palette=custom_palette)
```
![image](https://github.com/user-attachments/assets/949e599b-427c-4d1a-b1f8-7fc84a710f52)


2. 나이
여성의 구조율이 남성보다 월등히 높다는 것은 확인이 되었다. 이제 아이들에 대한 구조율이 어른보다 높은지 확인해 보겠다.
```
train_df.groupby(['Age', 'Survived'])['Survived'].count()
```
![image](https://github.com/user-attachments/assets/7954da53-71ca-4c87-bfa9-a122e09dd2ad)

kaggle에서 제공하는 데이터는 나이에 대한 자료가 위와 같이 굉장히 복잡하게 구성되어 있다. 따라서 분석에 앞서 이러한 나이를 구분하기 쉽게 정리하려고 한다. 특히, 나이에는 177개의 NaN 값이 포함되어 있다. 따라서 Age의 NaN 값 또한 해결해야할 문제이다. 필자는 Age에 있는 NaN 값들을 각 객실 등급의 평균 나이를 대상으로 구분하려고 한다. 그 이유는 다음과 같다. 1등실의 경우 부유한 귀족 계층이 탑승한 선실로 어느정도 나이가 있는 사람들이 많이 탑승하고 있다고 판단했기 때문이다. 반면, 3등실의 경우 가난한 사람들이 탑승한 선실로 아메리칸 드림을 꿈꾸고 타이타닉호의 마지막 정착지인 뉴욕 즉, 미국으로 향하는 사람들이 많았다고 판단하고 있다. 따라서 3등실의 경우 젊은 사람들이 많을 것으로 생각하고 있다. 2등실의 경우 중산층이 많은 선실로 1등실, 3등실의 중간으로 평균 나이 역시 중간으로 생각하고 NaN 값을 처리하려고 한다.
```
nan_age_df = train_df[train_df['Age'].isna()]
nan_counts_by_pclass = nan_age_df.groupby(['Pclass'])['PassengerId'].count()
nan_counts_by_pclass
```
1등실에는 30명, 2등실에는 11명, 3등실에는 136명의 승객들이 Age가 NaN 값이라는 것을 확인할 수 있다. 이들의 나이를 위에서 설명했던 방법을 토대로 각 선실 등급의 평균으로 대체하려고 한다.
```
average_pclass = train_df.groupby('Pclass')['Age'].mean()
train_df['Age'] = train_df.apply(lambda row: average_pclass[row['Pclass']] if pd.isna(row['Age']) else row['Age'], axis=1)
# NaN 값 확인
train_df['Age'].isna().sum()
```
Age에 대한 NaN 값을 처리했으니 연령대에 따라 구분을 해서 카테고리를 나누려고 한다. 기준은 현재 대한민국을 기준으로 했다. 당시 시대 상에 맞지 않다는 한계가 있지만 자료조사의 한계로 인해 현재 대한민국을 기준으로 구분하였다. 초등학교 입학 전까지를 Baby, 중학교 입학 전까지를 Child로 고등학교 졸업 전 즉, 고3까지를 Teenager로 구분하였다. 이후 남성 평균 대학 졸업 나이인 26까지를 Student로 그 이후부터 대한민국 통계청 자료에 따라 39세까지를 청년층(Young Adult)으로 구분했다. 이후 64세까지를 중장년층(Adult)로 그 이후는 노년층(Elderly)로 구분했다.
```
def get_category(age):
    cat = ''
    if age <= -1: cat = 'Unknown'
    elif age <= 8: cat = 'Baby'
    elif age <= 13: cat = 'Child'
    elif age <= 19: cat = 'Teenager'
    elif age <= 26: cat = 'Student'
    elif age <= 39: cat = 'Young Adult'
    elif age <= 64: cat = 'Adult'
    else: cat = 'Elderly'        
    return cat

group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Elderly']
 
train_df['Age_range'] = train_df['Age'].apply(lambda x : get_category(x))
predict_df['Age_range'] = predict_df['Age'].apply(lambda x : get_category(x))
```
연령대을 구분하고 나서 연령대별 선실 등급을 출력해보면 다음과 같다.
```
age_range_pclass = train_df.groupby(['Age_range', 'Pclass']).size().unstack()
age_range_pclass
```
![image](https://github.com/user-attachments/assets/4de00361-4d4c-4c34-860b-ee61bb6b7874)

NaN 값을 각 Pclass별 평균 나이로 대체했기 때문에 Young Adult, Adult의 값이 많은 것을 알 수 있다. 또한, 1등실에 Adult의 비율이, 3등실에 Young Adult가 많은 것을 알 수 있다. 뿐만 아니라 Baby, Child, Teenager의 수가 많은 것을 통해 가족 단위로 많이 탑승한 것을 알 수 있다. 다음으로 연령대별 생존자를 시각화하면 다음과 같다. 
```
plt.figure(figsize=(10,6))
sns.barplot(x='Age_range', y = 'Survived', hue='Sex', data=train_df, order=group_names)
```
![image](https://github.com/user-attachments/assets/d51fd634-b4a0-4eb2-9909-5e265e0bf12f)

Baby와 Child 부분에서의 여성, 남성 모두 구조율이 높은 것을 통해 아이들이 우선적으로 구조되었다는 것을 시각적으로 확인해 볼 수 있다. 뿐만 아니라 모든. 연령대에서 여성의 생존률이 높은 것을 통해 여성을 우선적으로 구조했다고 볼 수 있다. 하지만 Child에서 구조율이 낮은 이유는 다음과 같은 이유를 짐작해 볼 수 있다. 밑의 코드를 통해 Child일 경우 어떤 선실 등급에 속한지 확인해 보면 1등급실에 1명, 2등급실에 1명, 3등급실에 15명인 것을 알 수 있다. 따라서 대부분이 낮은 선실 등급에 속해 구조율이 낮았다고 볼 수 있다. 즉, 선실 등급이 구조율에 영향을 주었다고 볼 수 있다. 이후 Teenager부터 어린이로 취급되지 않기 때문에 남성의 구조율이 급격하게 낮아지는 것을 확인할 수 있다. 이런 점을 통해 연령대가 적당히 구분되었다는 것도 확인할 수 있다.
```
age_range_pclass_distribution = train_df.groupby(['Age_range', 'Pclass']).size().unstack()
child_pclass_distribution = age_range_pclass_distribution.loc['Child']
child_pclass_distribution
```


3. 사회적 지위
사회적 지위에 따른 비교는 Pclass(선실 등급)와 Fare(요금)을 통해 할 수 있다. 필자는 본문에서 Pclass를 먼저 분석해 보겠다. 지금까지 필자는 선실 등급이 높으면 구조율이 높다고 보고 있으며 지금까지 그 관점에 초점을 맞추고 EDA를 진행했다. 이번 파트에서 과연 그 추정이 맞는지 확인해 보고자 한다.
```
train_df['Pclass'].value_counts()
```
1등실에 184명, 2등실에 216명, 3등실에 491명으로 총 891명인 것을 알 수 있다. 다음으로 시각화를 통해 Pclass별 구조자가 얼마나 되는지 확인해 보고자 한다.

```
pclass = ["1", "2", "3"]
pclass_survived = {}
for i in pclass:
    total = train_df[train_df['Pclass'] == int(i)].shape[0]
    survived =  train_df[(train_df['Pclass'] == int(i)) & (train_df['Survived'] == 1)].shape[0]
    pclass_survived[i] = round(survived / total * 100, 2)

pclass_survived = pd.DataFrame.from_dict(pclass_survived, orient='index', columns=['Survival Rate (%)'])
pclass_survived = pclass_survived.reset_index()
pclass_survived.columns = ['Pclass', 'Survival Rate (%)']

custom_palette = ["#E6E6FA", "#FFA07A", "#AFEEEE"]
plt.figure()
ax = sns.barplot(x='Pclass', y='Survival Rate (%)', data=pclass_survived, palette=custom_palette)

for i, v in enumerate(pclass_survived['Survival Rate (%)']):
    ax.text(i, v, f"{v:.0f}%", color='black', ha='center', va='bottom', fontsize=10)

plt.title('Survival Rate by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Survival Rate (%)')
plt.legend()
plt.show()
```
![image](https://github.com/user-attachments/assets/66fe8a23-96fa-4935-9c7a-043e91f4dac4)

위의 시각화 결과를 통해 선실 등급이 높은 곳에 소속될 수록 구조율이 높은 것을 알 수 있다. 즉, 필자가 앞에서 전제로 보고 있었던 선실 등급이 높을 수록 구조율이 높다는 것이 맞다는 것을 확인할 수 있다. 다음으로 Fare(요금)에 따른 구조자도 보고자 한다. 요금에 따라 선실 등급에 차이가 발생하기 때문에 선실 등급에 따른 구조율을 비교해 보는데 좋은 데이터라고 생각하기 때문이다.

```
train_df.groupby(['Fare','Survived'])['Survived'].count()
```
![image](https://github.com/user-attachments/assets/1c1a6e7b-6cfd-44f7-a958-b42985e1cc81)

요금(Fare) 역시 Age와 비슷하게 굉장히 복잡하게 구성되어 있는 것을 알 수 있다. 따라서 연령대를 구분한 것처럼 요금 역시 구분을 지어야 할 필요가 있다. 당시 타이타닉호의 티켓 가격을 기준으로 1등석은 30파운드(150달러), 스위트 1등석은 870파운드(4350달러), 2등석은 12파운드(60달러), 3등실은 7파운드(35달러)으로 구분하려고 했으나 결과적으로 달러 구분했을 때 1등급실에 29명, 2등급실에 170명, 3등급실에 692명으로 위에서 Pclass별 통계와 맞지 않아 파운드로 구분했다.

```
def get_category(fare):
    cat = ''
    if fare >= 30: cat = 1
    elif fare >= 12: cat = 2
    else: cat = 3
    return cat

group_names = [1, 2, 3]
 
train_df['Fare_range'] = train_df['Fare'].apply(lambda x : get_category(x))
predict_df['Fare_range'] = predict_df['Fare'].apply(lambda x : get_category(x))

# 시각화
Fare_ranges = np.unique(train_df['Fare_range'].values)
Fare_range_survived = {}

for i in Fare_ranges:
    total = train_df[train_df['Fare_range'] == int(i)].shape[0]
    survived =  train_df[(train_df['Fare_range'] == int(i)) & (train_df['Survived'] == 1)].shape[0]
    if survived != 0:
        Fare_range_survived[i] = round(survived / total * 100, 2) 
    else:
        Fare_range_survived[i] = 0

fare_survived_df = pd.DataFrame.from_dict(Fare_range_survived, orient='index', columns=['Survival Rate (%)'])
fare_survived_df = fare_survived_df.reset_index()
fare_survived_df.columns = ['Fare_range', 'Survival Rate (%)']

custom_palette = ["#FF6B6B", "#FFD93D", "#9BDE7C"]
plt.figure(figsize=(10,6))
ax = sns.barplot(x='Fare_range', y='Survival Rate (%)', data=fare_survived_df, palette=custom_palette)

for i, v in enumerate(fare_survived_df['Survival Rate (%)']):
    ax.text(i, v, f"{v:.0f}%", color='black', ha='center', va='bottom', fontsize=10)

plt.title('Survival Rate by Fare Range')
plt.xlabel('Fare Range')
plt.ylabel('Survival Rate (%)')
plt.xticks(ticks=[0, 1, 2], labels=['1', '2', '3'], rotation=0)
plt.show()
```
![image](https://github.com/user-attachments/assets/126eb1c1-bb70-4ca4-9938-13e1562032ea)

위의 그래프처럼 요금에 따른 선실 등급이 높을 수록 구조율이 높은 것을 알 수 있다. 하지만 Fare Range를 통해 확인해 볼 수 있는 것이 하나있다. Fare range를 통해 구분한 1~3 등급이 과연 kaggle에서 제공한 Pclass와 맞을지에 대한 것이다.

```
train_df.groupby(['Pclass'])['Fare_range'].value_counts()
```
![image](https://github.com/user-attachments/assets/4fa88d46-47c8-4179-b97f-6baad1dfaf09)

Fare range를 통한 구분이 맞지 않다는 것을 확인할 수 있다. Pclass가 1인 반면 Fare range는 3으로 12파운드보다 적은 가격으로 구매했다는 것이다. 즉, Fare는 나중에 있을 모델을 통한 예측에서 오히려 헷갈림을 줄 수 있다. 따라서 Fare은 생존자를 예측하는 데에 있어 중요한 데이터로 보기 어렵다고 생각한다.


4. Embarked
```
print(train_df['Embarked'].count())
print(train_df[train_df['Embarked'].isna()])
```
Embarked의 개수와 중간 정착지에 따른 생존자의 숫자를 확인하면 다음과 같다. 총 개수는 889개로 train data에서 제공되는 891명의 승객과 두 명의 승객이 NaN 값으로 되어있다는 것을 알 수 있다. 위의 코드 중 마지막 코드를 통해 승객의 정보를 확인할 수 있다. 정보는 아래의 사진과 같다.
![image](https://github.com/user-attachments/assets/4b190b27-d054-421b-8a9f-c1dde7e3f324)

1등실 탑승객, 여성, 같은 선실(Cabin)에 탑승한 사람으로 둘 다. 구조된 사람이라는 것을 알 수 있다. 각 Embarked에 따른 생존율을 구하면 다음과 같다.

```
embarkeds = ["C", "Q", "S"]
embarked_survived = {}

for i in embarkeds:
    total = train_df[train_df['Embarked'] == i].shape[0]
    survived =  train_df[(train_df['Embarked'] == i) & (train_df['Survived'] == 1)].shape[0]
    embarked_survived[i] = round(survived / total * 100, 2)

for i in embarked_survived:
    print(f"{i} Embarked 생존률: {embarked_survived[i]}")


df_embarked_survived = pd.DataFrame.from_dict(embarked_survived, orient='index', columns=['Survival Rate (%)'])
df_embarked_survived = df_embarked_survived.reset_index()
df_embarked_survived.columns = ['Embarked', 'Survival Rate (%)']

custom_palette = ["#FF6B6B", "#FFD93D", "#9BDE7C"]
plt.figure()
ax = sns.barplot(x='Embarked', y='Survival Rate (%)', data=df_embarked_survived, palette=custom_palette)

for i, v in enumerate(df_embarked_survived['Survival Rate (%)']):
    ax.text(i, v, f"{v:.0f}%", color='black', ha='center', va='bottom', fontsize=10)

plt.title('Survival Rate by Embarked')
plt.xlabel('Embarked')
plt.ylabel('Survival Rate (%)')
plt.legend()
plt.show()
```
![image](https://github.com/user-attachments/assets/d437d92e-dc76-4671-b3ad-9d74a51b58f3)

타이타닉의 중간 정착지가 1. S = 영국 Southampton 2. C = 프랑스 Cherbourg, 3. Q = 아일랜드 Queenstown 4. 미국 New York 이렇게 되어 있다. 타이타닉호는 여정 중간에 내린 승객은 없었으며 모두가 New York으로 향할 예정이었다. 따라서 필자는 Embarked가 NaN인 두 승객은 여성, 1등실, 생존이라는 데이터를 이용해 생존 비율이 가장 높은 C인 Cherbourg로 임의로 채워 넣을 것이다.

```
train_df.loc[train_df['Embarked'].isna(), 'Embarked'] = 'C'
```
또한, Embarked는 중요한 feature로 볼 수 있다. 그 이유는 다음과 같다.
```
embarked_pclass_counts = train_df.groupby(['Embarked', 'Pclass']).size().unstack()
embarked_pclass_counts
```
![image](https://github.com/user-attachments/assets/97b6dae7-c8e7-43b0-b5fa-971eabfc5116)

위의 결과를 통해 승객들 중 영국에서 탑승한 승객이 가장 많았다. 뿐만 아니라 역사적으로 1910년 대 영국은 세계적으로 많은 사람들이 몰리던 곳이었다. 그 이유로는 세 가지가 있다.
1. 산업혁명과 경제적 기회
  * 18세기 영국은 산업혁명의 발상지로 세계 경제의 중심지로 영국으로 이주하거나 일자리를 찾기 위해 사람들이 몰리던 곳이었다.
2. 대영제국의 영향력
  * 영국은 당시 많은 식민지를 가지고 있었던 제국으로 중심지인 영국은 많은 사람들에게 기회의 땅이었다.
3. 이민과 무역의 중심지
  * 위의 두 가지 이유와 함께 영국은 당시 다른 대륙 사람들에게 이민 경유지였다. 즉, 북미, 오세아니아 등으로 이주할 때 영국을 거쳐서 갔다.
이러한 이유와 함께 당시 아메리칸 드림을 꿈꾸고 출발하는 3등실 승객이 영국에서 가장 많았을 것이라는 역사적 사실에 기반한 추측과 titanic data 분석을 통한 자료를 통해 사망자의 많은 비율이 영국 즉, Southampton에서 출발한 것을 확인할 수 있다. 따라서 필자는 Embarked를 주요한 feature로 생각하고 있다.


5. Cabin
Cabin에 대해 NaN 값을 확인해보면 687명이 선실 번호가 없다. 총 891명의 승객 중 687명의 데이터가 없는 것으로 필자는 Cabin feature은 삭제하기로 했다.
```
train_df[train_df['Cabin'].isna()]
```


6. SibSp, Parch
SibSp는 같이 탑승한 형제자매 또는 배우자 인원 수이며, Parch는 같이 탑승한 부모님 또는 어린이 인원 수이다. 이 두 feature가 생존에 영향을 주었는지는 모르지만 자료 조사 중 부부가 함께 탑승했을 경우 남성이 구명 보트에 탑승하지 못 하자 여성도 구명 보트에 탑승하지 않고 같이 배에서 최후를 맞이 했다는 내용이 있어 예상치 못한 영향을 줄 수 있다고 판단했다. 다음은 wikipedia에서 발췌한 내용이다.
> “노부부 스트라우스 부부는 금슬이 좋은 노부부였다. 이지도어 스트라우스가 구명보트 승선을 거절하자 그의 부인인 아이다 스트라우스도 선원의 구명보트 승선 제안을 거절했다.” - <https://ko.wikipedia.org/wiki/타이타닉호_침몰_사고>
즉, 이러한 예상하지 못한 부분에서 영향을 줄 수 있다고 판단해 제거하지 않고 분석해 보고자 한다.
```
train_df.groupby(['SibSp','Survived'])['Survived'].count()
train_df.groupby(['Parch','Survived'])['Survived'].count()
```
![image](https://github.com/user-attachments/assets/af117f8e-7b39-4945-be93-230cf74e401d)
![image](https://github.com/user-attachments/assets/fa92109f-6e13-4898-9b4e-eef612d0bfad)

위의 오른 쪽 결과를 통해 혼자 탑승한 경우, 배우자와 탑승한 경우, 배우자 및 형제자매와 함께 탑승한 승객의 숫자를 확인할 수 있다. 또한, Parch는 같이 탑승한 부모님 또는 어린이 인원 수이다. 또한, 왼 쪽 결과를 확인할 수 있다. SibSp, Parch는 분석하기 애매한 데이터이며, 위에서 소개한 혹시 모를 상황에 예상하지 못한 영향을 줄 수 있다고 판단한 부분이다. 마지막으로 필요 없다고 생각하는 feature은 제거하고 최종적으로 사용할 feature을 보면 다음과 같다.

```
def drop_features(df):
    df.drop(['PassengerId','Name', 'Ticket', 'Cabin', 'Age', 'Fare', 'Fare_range'], axis=1, inplace=True)

    y = df['Survived']
    df = df.drop('Survived', axis=1, inplace=False)
    return df, y
    
X, y = drop_features(train_df)
feature = X.columns
predict_df = predict_df[feature]
```
![image](https://github.com/user-attachments/assets/160321bb-8be3-4afb-b5d7-e341d75848ed)








