import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
from PIL import Image


def get_user_data():
    st.sidebar.title("User Prediction Inputs")
    Pregnancies = st.sidebar.slider('Pregnancies',0,20,3)
    Glucose= st.sidebar.slider('Glucose',0,200,110)
    BloodPressure= st.sidebar.slider('BloodPressure',0,120,60)
    SkinThickness= st.sidebar.slider('SkinThickness',0,100,20)
    Insulin= st.sidebar.slider('Insulin',0.0,900.0,30.50)
    BMI= st.sidebar.slider('BMI',0.00,70.0,32.0)
    DiabetesPedigreeFunction= st.sidebar.slider('DiabetesPedigreeFunction',0.078,2.50,0.3000)
    Age=st.sidebar.slider('Age',0,100,20)

    user_data = {'Pregnancies' : Pregnancies,
                 'Glucose':Glucose,
                 'BloodPressure':BloodPressure,
                 'SkinThickness':SkinThickness,
                 'Insulin':Insulin,
                 'BMI':BMI,
                 'DiabetesPedigreeFunction':DiabetesPedigreeFunction,
                 'Age':Age}
    #Transform data into dataframe
    features=pd.DataFrame(user_data,index=[0])

    return features


#Title of web app
st.write("""
# Diabetes Detector
Use Machine Learning to predict Diabetes!
""")

#Web Image
img = Image.open("Background deisgn.jpg")
st.image(img,use_column_width=True)

#Load data
df = pd.read_csv("diabetes.csv")
st.subheader("Diabetes Data Information:")
st.dataframe(df) #Display the dataframe
st.write(df.describe())#Display the gist of the data
st.bar_chart(df)#Display a bar chart of data

#Data processing
X = df.iloc[:,0:8] #dependent variables
y = df.iloc[:,-1] #independentt variables

#split to train and test 80:20 ratio

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Get inputs from user for prediction
user_input = get_user_data()

#Sub heading for user input
st.subheader("User Inputs ")
st.write(user_input)

#Train the model using random forest
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train,y_train)

#Show model metrics
st.subheader("Model Test Accuracy: ")
st.write(f"{accuracy_score(y_test,RandomForestClassifier.predict(X_test))*100}%")

#Store predictions
prediction = RandomForestClassifier.predict(user_input)
output = "No Diabetes Detected !"
if prediction==1:
    output = "Diabetes Detected !"

#Sub header for prediction
st.subheader("Prediction: ")
st.write(output)





