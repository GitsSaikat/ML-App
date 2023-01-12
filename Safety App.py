from operator import index
import streamlit as st
import pickle
import plotly.express as px
from pycaret.classification import *
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import os 

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("Women T Safety Logo.jpg")
    st.title("Machine Learning Based Application on Women Transportation Safety")
    choice = st.radio("Navigation", ["Homepage","Upload","Profiling","Modelling", "Download"])
    st.info("User can Upload their Dataset(in Upload), Visualize the Profiling of their Dataset(in Profiling Option), Train the Dataset with Radom Forest Classifier and Visualize the Model's performance(in Modelling Option), And Download the model (in Download Option)")

if choice=="Homepage":

    st.title("Homepage")
    st.image("Happy Women.jpg")
    st.write("Women's transportation safety is a significant and ongoing problem that has received increasing attention in recent years. According to a report by the World Health Organization, women and girls face unique security challenges while using transportation services, including sexual harassment, violence, and discrimination. These risks are particularly prevalent in low-income and developing countries, where public transportation may be poorly regulated and infrastructure may be inadequate.")
    st.write("To address the problem of women's transportation safety, it is necessary to adopt a range of strategies and interventions that address the root causes of violence and harassment against women and that ensure the safety and well-being of women while using transportation services. These strategies may include increasing security measures, regulating rideshare and taxi services, and providing education and awareness campaigns to educate women about their rights and the resources available to them in an emergency. It is also important to prioritize the safety and well-being of women when designing and implementing transportation systems and services.")
    

    st.write("This application helps Explore dataset on Women Transportation Safety and Build Model and Visualize the Model's Performance. User can also the download the model in pkl format")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Modelling":
    st.title("Build the Model") 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        exp_name = setup(data = df,  target = chosen_target)
        rf = create_model('rf')
        save_model(rf, 'Model')
        plot_model(rf, plot ='class_report', display_format= 'streamlit')
        plot_model(rf, plot ='confusion_matrix', display_format= 'streamlit')
        plot_model(rf, plot ='pr', display_format= 'streamlit')
        plot_model(rf, plot ='auc', display_format= 'streamlit')


if choice == "Download": 
    st.title("Download the Model")
    st.text("Press Download Button to Download the Model")
    with open('Model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="Model.pkl")
        