import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pycaret.classification import *
import shap
import matplotlib.pyplot as plt

#st.markdown('<p align="center"> Water Quality **Estimation** </p>', unsafe_allow_html=True))

st.markdown(' <p align="center" class="big-font">  <b>Water Quality <u>Check</b>   </p>', unsafe_allow_html=True)	


st.markdown("""
<style>
.big-font {
    font-size:45px !important;
}
</style>
""", unsafe_allow_html=True)

#st.markdown('<p class="big-font">Hello World !!</p>', unsafe_allow_html=True)
 
st.markdown(' <p align="center"><img width="523" src="https://user-images.githubusercontent.com/20365333/140042600-a602ed75-6571-4f7b-adee-0d33d51f9cf0.jpg"></p>', unsafe_allow_html=True)	

#st.write(""" """)

st.markdown("""
	AKN is a material testing laboratory that focuses on water quality tests and classification. Our next step,
	to expand the business research and development that to start a project to automate the water classification process.
	Access to safe drinking-water is essential to health, a basic human right and a component of effective policy for health protection.

	""")

st.subheader("sample values for the input")
df=pd.read_csv("example.csv")
#df.drop('Unnamed: 0', axis=1, inplace=True)
df

#df_example=df.iloc[df['ph']==6.007427,['Organic_carbon','Conductivity','Hardness']]
#df_example=df.iloc[0,1:]
#df_example

if st.checkbox("Show orignal dataframe"):
	dataframe=pd.read_csv("water1.csv")
	dataframe.drop('Unnamed: 0', axis=1, inplace=True)
	dataframe

##Sidebar

st.sidebar.title("Select your Input  Values")

uploaded_file=st.sidebar.file_uploader("Upload your csv file in the same input as the example csv file",type=["csv"])

if uploaded_file is not None:
	input_params=pd.read_csv(uploaded_file)

else:
	ph=st.sidebar.slider("ph value",0.1,28.3,7.5)
	Hardness=st.sidebar.slider("Hardness value",47.432,323.3,118.2)
	Solids=st.sidebar.slider("Solids value(mg/L) ",181.4,30000.0,14285.58)
	Chloramines=st.sidebar.slider("Chloramines value",0.1,28.3,9.27)
	Sulfate=st.sidebar.slider("Sulfate value(mg/L)",47.432,400.3,333.07)
	Conductivity=st.sidebar.slider("Conductivity value(μS/cm)",181.4,753.2,418.60)
	Organic_carbon=st.sidebar.slider("Organic Carbon value(mg/L)",2.1,28.3,16.86)
	Trihalomethanes=st.sidebar.slider("Trihalomethanesvalue(ppm)",47.432,323.3,66.42)
	Turbidity=st.sidebar.slider("Turbidity value(NTU)",0.1,28.3,3.05)

	dict_values={"ph":ph, "Hardness":Hardness, "Solids":Solids,"Chloramines":Chloramines,"Sulfate":Sulfate,
		     "Conductivity":Conductivity,"Organic_carbon":Organic_carbon,
		     "Trihalomethanes":Trihalomethanes,"Turbidity":Turbidity}
	features=pd.DataFrame(dict_values,index=[0])
	input_params=features



#ph Solids Chloramines Sulfate Trihalomethanes Turbidity

st.subheader("user input fields")

if uploaded_file is not None:
	st.write(input_params)
else:
	st.write(input_params)

#load_clf=pickle.load(open('dt_saved_07032020.pkl','rb'))
#load_clf= pd.read_csv("water1.csv")
load_clf= load_model('dt_saved_07032020')
prediction=load_clf.predict(input_params)
st.write('---')
st.subheader(":hourglass: The Prediction is")
#st.write()
st.write(""":bulb:==> """,prediction[0])
if(prediction[0]==1):
	st.subheader("The water is safe to drink :droplet: :thumbsup:")
else:
	st.subheader("The water is not safe to drink :warning: :skull:")
st.write('* 0 = not safe, 1= safe')
