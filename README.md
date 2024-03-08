# KHUDA 경희대학교 국제캠퍼스 맛집 추천 개발

### Table of Contents
* [General info](#general-info)
* [Utilized Model & Skill](#UtilizedModel&Skill)
* [Problem-solving process](#Problem-solvingProcess)
* [Members](#Members)


## How to Use
1. Clone the repository :
```
git clone https://github.com/BARAM1NG/FoodRec.git
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Start the development server:
```
streamlit run app.py  
```

## Utilized Model & Skill
- MLP (Reliability)
- Streamlit (Web demo)
- Google Map API
- ~~KNN~~
- ~~Recommendation Algorithm~~

## Problem-solving process
- We have devised an idea to reassess the reliability of restaurants from the perspective of students at Kyung Hee University, judging that the restaurant ratings on Google Maps are not reliable due to advertising jobs.
- In the process of developing an MLP model, we're incorporating inputs such as food type, price, Google Maps ratings, visitor reviews, and blog reviews to determine the trustworthiness (restaurant score) of each establishment.
- We attempted to create a recommendation algorithm using KNN, but encountered difficulties in the specific algorithm creation process. As a result, We opted to develop an algorithm without utilizing KNN.
  - Continuing attempts in progress...

## Web Demo Screen

<img width="1876" alt="Picture_3" src="https://github.com/BARAM1NG/FoodRec/assets/122276734/6ebf7315-e040-4de8-820d-c3ec081cb214">
<img width="1876" alt="Picture_2" src="https://github.com/BARAM1NG/FoodRec/assets/122276734/659e78c6-74f4-4a57-960b-d20d4718b824">
<img width="1875" alt="Picture_1" src="https://github.com/BARAM1NG/FoodRec/assets/122276734/cdf62827-05cc-4a49-9705-3387dd505c5e">



## Our Project Archive
This is our notion page for our project archive. : 
[Notion](https://baram1ng.notion.site/KHUDA-TOY-PROJECT-61b9175a3aee4a4a8df822bcf81ce19b?pvs=4)

## Members
|멤버이름|역할|
|------|---|
|배아람|Leader, Data Collection, Data Modeling, Web Demo(streamlit)|
|이하영|Data Collection, Outline of Model Code, Data Modeling|
|박성수|Data Collection, Data Modeling|
|정소연|Data Collection, Data Modeling|
|박재환|Data Collection, Data Modeling|
|장유리|Data Collection, Data Modeling|
|윤소은|Data Collection, Data Modeling|
