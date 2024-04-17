import pandas as pd
import streamlit as st 
st.header("Profit Prediction")
st.sidebar.header("Options")
Spend=st.sidebar.slider("R&D_Spend",15,100,66)
Administration=st.sidebar.slider("Administration",85,110,95)
Marketing_Spend=st.sidebar.slider("Marketing_Spend",88,120,95)


df=pd.read_csv("App/Startups.csv")
x=df.iloc[: ,:-1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict([[Spend,Administration,Marketing_Spend]])
st.subheader("Predicted Profit")
st.write(y_pred)
from sklearn.metrics import r2_score
y_pred1=lr.predict(x_test)
r2=r2_score(y_test,y_pred1)
st.subheader("R2 score")
st.write(r2)