import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

## Load the dataset
st.sidebar.header("Upload your CSV file")
uploadedFile = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploadedFile is not None:
    data = pd.read_csv(uploadedFile)

    st.title("Titanic Survivors Prediction APP")

    ## Show raw dataset
    if "data" in locals() or "data" in globals():
        st.header("Raw Dataset")
        isCheck = st.checkbox("Show raw dataset")
        if isCheck:
            st.write(data)

    ## Feature Selection
    # 열의 이름을 제거한 후 리스트로 반환
    importantFeatures = data.columns.drop(["PassengerId", "Name", "Ticket",
                                           "Cabin", "Survived"]).tolist()
    
    ## Selecting Features for Prediction
    st.header("Select Features for Prediction")
    # multiselect(설명텍스트, 선택가능한 피처의 전체목록, 기본적으로 선택될 피처)
    selectedFeatures = st.multiselect("Select features to use for prediction:",
                                      options = importantFeatures, default=
                                      ["Pclass", "Sex", "Age", "Fare", "Embarked"])
    # 현재 선택된 피처를 세션 상태로 저장
    if "selectedFeatures" not in st.session_state:
        st.session_state["selectedFeatures"] = selectedFeatures
    # 피처가 바뀌면 모델을 다시 학습하도록 설정
    if selectedFeatures != st.session_state['selectedFeatures']:
        del st.session_state["gridBestModel"]
        st.session_state["selectedFeatures"] = selectedFeatures

    ## Data Preprocessing
    data = data[selectedFeatures + ["Survived"]].dropna()
    data = pd.set_dummies(data, drop_first=True)
    st.write(data)

    ## splitting the Data
    X = data.drop("Survived", axis=1)
    y = data["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42)
    
    ## Hyperparameter Tuning using GridSearchCV
    st.header("Hyperparameter Tuning with GridSearchCV")
    paramGrid = {
        "n_estimators" : [50, 100, 150],
        #None 깊이 제한 없음
        "max_depth" : [None, 10, 20],
        "min_samples_split" : [2, 5, 10],
        "min_samples_leaf" : [1, 2, 4]
    }
    #학습된 모델을 저장할 수 있는 세션 상태 확인
    if "gridBestModel" not in st.session_state:
        # n_jobs=-1 병렬처리, -1은 가능한 모든 코어 동원
        # verbose : GredSearchCV가 수행하는 각 단계에 대한 진행 상황을 콘솔에 출력
        gridSearch = GridSearchCV(RandomForestClassifier(random_state=42),
                                  paramGrid, cv=3, n_jobs=-1, verbose=2)
        gridSearch.fit(X_train, y_train)
        st.session_state["gridBestModel"] = gridSearch.best_estimator_
        # st.write(st.session_state)

    ## Best model
    gridBestModel = st.session_state["gridBestModel"]
    # st.write(gridBestModel)

    ## Making predictions
    y_pred = gridBestModel.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.wirte(f"Accuracy of the best model: {accuracy * 100:.2f}%")

    ##Feature Importance Analysis
    st.header("Feature Importance")
    featureImportances = np.sort(gridBestModel.feature_importances_)[::-1]
    featureImportances = featureImportances.reshape(1,-1)
    featureImportances = pd.DataFrame(featureImportances,
                                      columns=X.columns)
    
    ## Plotting Feature Importance
    fig, ax = plt.subplot()
    sns.barplot(x=featureImportances.iloc[0, :],
                y=featureImportances.columns, ax=ax)
    ax.set_title("Feature Importance")
    ax.set_ylabel("Feature")
    ax.set_xlabel("Importance")
    st.pyplot(fig)

    ## User Input for Prediction
    st.header('Predict Survival')
    userInput = {}
    for feature in selectedFeatures:
        if feature == "Pclass":
            userInput[feature] = st.radio(
                "Select Pclass",
                options=[1,2,3],
                index=0,
                horizontal=True
            )
        elif feature in ['Fare', 'Age']:
            userInput[feature] = st.slider(
                f"Enter {feature}",
                min_value=float(data[feature].min()),
                max_value=float(data[feature].max()),
                value=float(data[feature].mean()),
            )
        elif feature in ['SibSp', 'Parch']:
            userInput[feature] = st.slider(
                f"Enter {feature}",
                min_value=float(data[feature].min()),
                max_value=float(data[feature].max()),
                value=float(data[feature].mean()),
            )
        elif feature == "Sex":
            # Male이면 true, Female이면 false
            userInput["Sex_male"] = st.selectbox("Select Gender",
                                                 options=['Male', 'Female']) == 'Male'
        elif feature == "Embarke":
            userInput[feature] = st.selectbox("Select Embarkation Port",
                                                 options=['C', 'Q', 'S'])
            if userInput[feature] =='Q':
                userInput["Embarked_Q"] = 1
                userInput["Embarked_S"] = 0
            elif userInput[feature] =='S':
                userInput["Embarked_Q"] = 0
                userInput["Embarked_S"] = 1
            else:
                userInput["Embarked_Q"] = 0
                userInput["Embarked_S"] = 0

    ## Convert User Input to DataFrame
    inputDF = pd.DataFrame([userInput])
    if "Embarked" in inputDF.columns:
        inputDF = inputDF.drop(columns=["Embarked"], axis=1, errors='ignore')

    ##Convert inputDF to numpyArray and convert 1D array to 2D
    inputArray = np.array(inputDF.iloc[0, :])
    inputArray = inputArray.reshape(1, -1)

    ## Prediction Button
    if st.button('Predict'):
        prediction = gridBestModel.predict(inputArray)[0]
        if prediction == 1:
            st.success('The passenger is likely to survive!')
        else:
            st.error('The passenger is unlikely to survive')
        
else:
    st.error("No dataset available. Please upload a CSV file.")
        