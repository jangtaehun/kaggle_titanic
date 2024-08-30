### Titanic - Machine Learning from Disaster
kaggle에서 제공하는 Tatanic data를 이용해 EDA와 model 학습을 통해 생존자를 예측하는 프로젝트

---
### 분석 기간
2024.08.20 - 2024.08.21

---

### 소개
타이타닉은 세계에서 가장 유명한 침몰선이라 할 수 있으며, 사고로부터 100년이 넘게 지난 오늘날까지도 관련 연구가 활발하게 이루어지며 대중매체 등에서 많이 다뤄지고 있습니다. 그래서 많은 사람들이 머신러닝 학습을 처음 시작할 때 Kaggle에서 제공하는 타이타닉 데이터를 이용해 Kaggle 자체 대회에 참여하고 있습니다.
따라서 필자 역시 타이타닉 데이터를 이용한 Kaggle 타이타닉 대회에 참여해 가능한 높은 점수를 얻어보고자 이 프로젝트를 시작하게 되었습니다.

---


### 프로젝트 개요
##### 목표
이 프로젝트의 목표는 타이타닉 탑승객의 생존 여부를 다양한 특징들을 바탕으로 예측하는 것입니다. 데이터셋은 탑승객의 인구통계, 티켓 등급, 선실 정보 등을 포함하고 있습니다. 이러한 특징들을 분석함으로써, 탑승객의 생존 여부를 정확하게 예측하는 모델을 개발하고자 합니다.

##### 데이터셋 (Data Set)
이 프로젝트에서 사용한 데이터셋은 Kaggle에서 제공하는 다음 파일들로 구성되어 있습니다.
1. titanic_train.csv: 훈련 데이터셋, 특징들과 목표 변수(Survived)를 포함.
2. test.csv: 테스트 데이터셋, 예측을 위해 사용될 데이터.
3. gender_submission.csv: 예측 결과를 제출하기 위한 샘플 파일.

---

##### 방법론
 프로젝트는 다음과 같은 순서로 진행됩니다.
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
 
  * ___ 승무원의 경우 갑판부, 기관부, 사주부가 있으나 Kaggle에서 제공하는 data set에서는 승무원들에 대한 정보가 없기 때문에 생략하겠다. ___

* 충돌 및 탈출
 * 우현측면이 빙산과 충돌
 * 선원들이 여자와 아이들을 먼저 태울 것을 건의했으며, 선장은 승인했다. 하지만 소통의 오류로 ‘여성과 아이들만’으로 전달되어 여성과 어린이만 태웠기 때문에 자리가 있었음에도 남자는 승무원들이 거부해 구명보트 정원의 절반도 못 태운채 보트가 있었다. 뿐만 아니라 1,178명 정도를 태울 수 있는 구명보트만 구비되어 있었기 때문에 큰 인명 피해가 발생했다.
