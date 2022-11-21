from operator import index
import streamlit as st
import plotly.express as px
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 
import numpy as np
import re 
from pydantic.typing import NoneType
from ast import literal_eval
st.set_page_config(layout="wide")
def get_parameters(func):
    keys = func.__code__.co_varnames[:func.__code__.co_argcount][::-1]
    sorter = {j: i for i, j in enumerate(keys[::-1])} 
    values = func.__defaults__[::-1]
    kwargs = {i: j for i, j in zip(keys, values)}
    sorted_args = tuple(
        sorted([i for i in keys if i not in kwargs], key=sorter.get)
    )
    sorted_kwargs = {
        i: kwargs[i] for i in sorted(kwargs.keys(), key=sorter.get)
    }   
    return sorted_args, sorted_kwargs

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoML")
    choice = st.radio("Navigation", ["Upload","Profiling","Modeling", "Download"])
    st.info("This project application helps you build and explore your data.")

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

if choice == "Modeling": 
    st.title("Train your Model")
    # Format the target columns with their dtypes
    lst_columns=[]
    for i in range(0,len(df.columns)):
        lst_columns.append(df.columns[i] + f' ({str(np.array(df.dtypes)[i])})')
    chosen_target = st.selectbox('Choose the Target Column', lst_columns)
    chosen_target = re.sub("[\(\[].*?[\)\]]", "", chosen_target).strip()
    numerical_features = df.select_dtypes(include=['int64','float64'])
    categorical_features = df.select_dtypes(exclude=['int64','float64']) 
    # Find if it is a Regression or Classification Problem using their dtypes
    if str(np.array(df[chosen_target].dtypes)) in list(categorical_features.dtypes.unique()):
        from pycaret.classification import *
        st.write("This is a Classification Problem")
    else:
        from pycaret.regression import *
        st.write("This is a Regression Problem")
    # Write all the parameters for the fine tuning of the model
    col1, col2 = st.columns((1,2))
    from streamlit.components.v1 import html
    with col1 :
        if st.checkbox('Fine Tuning Model'):
            if st.checkbox("Help"):
                #import pydoc
                #strhelp = pydoc.render_doc(setup, "Help on %s")
               
                embed_help= ("""<object type="text/html" data="https://pycaret.readthedocs.io/en/stable/api/classification.html" height=500 width=350</object>""")
                html(embed_help, height=500, scrolling=True) #st.write(setup)
            
            st.subheader("Model Parameters")
            sorted_args, sorted_kwargs = get_parameters(setup)
            sorted_params = list(sorted_kwargs.keys())
            sorted_params_values = list(sorted_kwargs.values()) 
            lst_parms=[]
            lst_bool=[]  
            for i in range (0,len(sorted_params)):
                
                if type(sorted_params_values[i]) is str:
                    text_inpt = st.text_input(sorted_params[i],value=sorted_params_values[i],
                    placeholder=sorted_params_values[i],
                    )
                    lst_parms.extend([[text_inpt,i]])
                    
                        
                elif type(sorted_params_values[i]) is NoneType:
                    sorted_params_values[i] = None
                    none_type_inpt = st.text_input(label=sorted_params[i],value=sorted_params_values[i])
                    lst_parms.extend([[none_type_inpt,i]])
                    
                        

                elif type(sorted_params_values[i]) is float or int:
                    #   if type(sorted_params_values[i]) is bool :
                    #    bool_inpt = st.text_input(sorted_params[i],value=sorted_params_values[i],
                    #placeholder=sorted_params_values[i],
                    #)
                    #    lst_parms.extend([[bool_inpt,i]])

                    numbers_inpt = st.number_input(label=sorted_params[i],value=sorted_params_values[i])
                    lst_parms.extend([[numbers_inpt,i]])

                    
                    # append the values in a list
                


            result = {}
            lst_parms = [i[0] for i in lst_parms]
            lst_parms = [float(x) if str(x).replace('.', '').isdigit() and float(x) <=1 else x for x in lst_parms]
            lst_parms = [None if str(x) == "None" else x for x in lst_parms]
            lst_parms = [True if  x == 1.0 else False if x == 0.0 else x for x in lst_parms]
            for params, values in zip(sorted_params,lst_parms):
                result.setdefault(params,values)
            if result["silent"] == False :
                result["silent"] = True
            
            print(lst_parms)
            print(result)
            
    with col2:
        if st.button('Run Modeling'): 
                    st.write("Modeling ...")
                    setup(data = df, target=chosen_target,**result)
                    setup_df = pull()
                    st.dataframe(setup_df)
                    best_model = compare_models()
                    compare_df = pull()
                    st.dataframe(compare_df)
                    save_model(best_model, 'best_model')
                    st.write("Done")

    

if choice == "Download": 
    st.title("Download the Model")
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")
