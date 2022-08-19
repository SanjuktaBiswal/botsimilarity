import streamlit as st
from state import provide_state
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.models import DataTable, TableColumn, HTMLTemplateFormatter
from streamlit_bokeh_events import streamlit_bokeh_events
import streamlit.components.v1 as components
from master_usecase_similarity_index import initialise,query_op
from BotDesc_Similarity_index import queryprocessing
from string_master import string_search_master
from string_Bot import string_search_bot
st.set_page_config(layout="wide")
header=st.beta_container()

features=st.beta_container()
df_result=None
st.markdown(
"""
<style>.main
{
background-color:#F5F5F5;
}
</style>
""",unsafe_allow_html=True
)
background_color='#F5F5F5'
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

def main():
    header=st.beta_container()
    
    local_css("style.css")
    #Smenu=["Search"]
    placeholder = None
    placeholder_search = None
    placeholder_col1_display = None
    placeholder_col2_display = None
    placeholder_col1_test=None
    #choice=st.sidebar.selectbox("Menu",menu)
    # A variable to keep track of which product we are currently displaying
    if 'counter_one' not in st.session_state:
        st.session_state.counter_one=0
    if 'counter_two' not in st.session_state:
        st.session_state.counter_two=pd.DataFrame()
    if 'radio' not in st.session_state:
        st.session_state.radio="Search In Master Usecase Table"
    if 'radiobutton_search' not in st.session_state:
        st.session_state.radiobutton_search=""
    counter=0
    #if choice=="Search":
    st.title("Search")
    components.html("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """)
    with st.form(key="similarity"):
        col1,col2,col3,col4=st.beta_columns([2,3,0.5,1])
        with col1:
            button_names=["Search In Master Usecase Table","Search In Bot Description Table"]
            radiobutton=st.radio("Navigation",button_names)
        with col2:
            
            placeholder_search = st.empty()
            
            sentence = placeholder_search.text_input('VM Server Decomission')
        
            
        with col4:
            placeholder_semantic_string = st.empty()
            
            button_search=["Semantic Search","String Search"]
            st.session_state.radiobutton_search=placeholder_semantic_string.radio("Navigation",button_search)
            
            
            flag=st.form_submit_button('Search')
   
        if flag:    
                    st.session_state.radio=radiobutton
                    
                    placeholder = st.empty() 
                    if len(sentence)==0:
                        st.session_state.counter_two=pd.DataFrame()
                        placeholder.warning('Kindly Enter Search Criteria')
                    elif len(sentence)>0:
                        msg=placeholder.text('Fetching data. Kindly Wait...')
                        
                        
                        try:
                            if st.session_state.radio=="Search In Bot Description Table":
                                if st.session_state.radiobutton_search=="Semantic Search":
                                
                                    df,df_result=queryprocessing(sentence.strip().lower())
                                    
                                    if df_result.shape[0]>0:
                                            msg.success('Success...')
                                            st.session_state.counter_two=df.copy()                                   
                                elif st.session_state.radiobutton_search=="String Search":
                                    df,df_result=string_search_bot(sentence.strip().lower(),"data/Deployment.xlsx")
                                    
                                    if df_result.shape[0]>0:
                                            msg.success('Success...')
                                            st.session_state.counter_two=df.copy() 
                                            
                                            
                            elif st.session_state.radio=="Search In Master Usecase Table":
                                if st.session_state.radiobutton_search=="Semantic Search":
                                    
                                    embedding_dict,df,data=initialise()
                                    df_result=query_op(embedding_dict,df,data,sentence.strip().lower())
                                    
                                    if df_result.shape[0]>0:
                                            msg.success('Success...')
                                            st.session_state.counter_two=df_result.copy() 
                                elif st.session_state.radiobutton_search=="String Search":
                                    
                                    
                                    df_result=string_search_master(sentence.strip().lower(),"data/Deployment.xlsx")
                                    
                                    if df_result.shape[0]>0:
                                            msg.success('Success...')
                                            st.session_state.counter_two=df_result.copy()        
                                    
                        except KeyError as  e:
                            st.session_state.counter_two=pd.DataFrame()
                            placeholder.error('Returned 0 records.\nKindly Check Search Criteria')           
                        except Exception as  e:
                            st.session_state.counter_two=pd.DataFrame()
                            st.write(e)
    if (st.session_state.counter_two.shape[0]>0):
        if st.session_state.radio=="Search In Bot Description Table":
                            if st.session_state.radiobutton_search=="Semantic Search":
                                
                                st.subheader("Search Results")
                                st.text('\n')
                          
                                
                                data=pd.DataFrame()                  
                                
                                data=st.session_state.counter_two.copy()
                                
                                data=data.loc[:,["BoT Name","Bot Desc","DemandID"]]
                                st.session_state.counter_two.index=np.arange(1, len(st.session_state.counter_two) + 1)
                                data.index = np.arange(1, len(data) + 1)
                                #st.write(data.head(10)) 
                                fig=go.Figure(data=go.Table(
                                columnwidth=[0.5,0.75,1.25,0.5],
                                header=dict(values=list(["Row No.","BoT Name","Bot Desc","Demand ID"]),
                                fill_color='#FD8E72',align='center'),cells=dict(values=([st.session_state.counter_two.index[:],st.session_state.counter_two["BoT Name"][0:].tolist(),st.session_state.counter_two["Bot Desc"][0:],st.session_state.counter_two["DemandID"][0:].tolist()]))))
                    
                                fig.update_layout(width=1000,height=350,margin=dict(l=1,r=1,b=15,t=15),
                                paper_bgcolor = background_color
                                
                                )
                                st.write(fig)  
                            else:
                                
                                st.subheader("Search Results")
                                st.text('\n')
                          
                                
                                data=pd.DataFrame()                  
                                
                                data=st.session_state.counter_two.copy()
                                
                                data=data.loc[:,["DemandID","BoTID","clean_Bot Desc"]]
                                st.session_state.counter_two.index=np.arange(1, len(st.session_state.counter_two) + 1)
                                data.index = np.arange(1, len(data) + 1)
                                #st.write(data.head(10)) 
                                fig=go.Figure(data=go.Table(
                                columnwidth=[0.5,0.5,0.5,1.5],
                                header=dict(values=list(["Row No.","Demand ID","BoT ID","Bot Description"]),
                                fill_color='#FD8E72',align='center'),cells=dict(values=([st.session_state.counter_two.index[:],st.session_state.counter_two["DemandID"][0:].tolist(),st.session_state.counter_two["BoTID"][0:],st.session_state.counter_two["clean_Bot Desc"][0:].tolist()]))))
                    
                                fig.update_layout(width=1000,height=350,margin=dict(l=1,r=1,b=15,t=15),
                                paper_bgcolor = background_color
                                
                                )
                                st.write(fig)  
        else:
            
            with st.form(key="result"):
                             
                    try:
                        
                        if st.session_state.radio=="Search In Master Usecase Table":
                            if st.session_state.radiobutton_search=="Semantic Search":
                            
                                    col1,col2 = st.beta_columns(2)
                                    with col1:
                                            st.subheader("Search Results")
                                            st.text("Click on any \"Deployment ID\" in the first Columns of the\nsearch Results Table to View the Microbots associated with\nthat Deployment ID")
                                            st.text('\n')
                                            data=pd.DataFrame()                  
                                            
                                            data=st.session_state.counter_two.copy()
                                            
                                            data=data.loc[:,["DeploymentID","AccountName","Similarity_index"]]
                                            st.session_state.counter_two.index=np.arange(1, len(st.session_state.counter_two) + 1)
                                            data.index = np.arange(1, len(data) + 1)
                                            #st.write(data.head(10)) 
                                            fig=go.Figure(data=go.Table(
                                            columnwidth=[0.5,0.75,1],
                                            header=dict(values=list(["Row No.","Deployment ID","Account Name","Similarity Index"]),
                                            fill_color='#FD8E72',align='center'),cells=dict(values=([st.session_state.counter_two.index[:],st.session_state.counter_two["DeploymentID"][0:].tolist(),st.session_state.counter_two["AccountName"][0:].tolist(),st.session_state.counter_two["Similarity_index"][0:].tolist()]))))
                                
                                            fig.update_layout(width=500,height=350,margin=dict(l=1,r=1,b=10,t=10),
                                            paper_bgcolor = background_color
                                            
                                            )
                                            st.write(fig)
                                    with col2:
                                        if st.session_state.radio=="Search In Master Usecase Table":
                                            
                                            placeholder_result = st.empty()
                                            result = placeholder_result.text_input('Row Number')
                                            st.session_state.counter_three=placeholder_result  
                                            btn=st.form_submit_button('Fetch')
                                            if btn:
                                                try:
                                                    
                                                        
                                                    if isinstance(int(result.strip()), int):
                                                        df=pd.DataFrame(st.session_state.counter_two.loc[int(result)]).T
                                                        
                                                        st.dataframe(df.style.set_properties(**{'background-color': '#FFA500',
                                                               'color': 'white'}))
                                                        #st.write(pd.DataFrame(st.session_state.counter_two.loc[int(result)]).T)
                                                    
                                                    
                                                except Exception as e:
                                                     st.error(e)
                                                     #placeholder_result.text_input('Row Number',value='')
                                                     placeholder_result.text_input('Row Number',value='')
                            if st.session_state.radiobutton_search=="String Search":
                            
                                    col1,col2 = st.beta_columns(2)
                                    with col1:
                                            st.subheader("Search Results")
                                            st.text("Click on any \"Deployment ID\" in the first Columns of the\nsearch Results Table to View the Microbots associated with\nthat Deployment ID")
                                            st.text('\n')
                                            data=pd.DataFrame()                  
                                            
                                            data=st.session_state.counter_two.copy()
                                            
                                            data=data.loc[:,["DeploymentID","clean_MasterUseCase"]]
                                            st.session_state.counter_two.index=np.arange(1, len(st.session_state.counter_two) + 1)
                                            data.index = np.arange(1, len(data) + 1)
                                            #st.write(data.head(10)) 
                                            fig=go.Figure(data=go.Table(
                                            columnwidth=[0.5,0.75,1],
                                            header=dict(values=list(["Row No.","Deployment ID","MasterUseCase"]),
                                            fill_color='#FD8E72',align='center'),cells=dict(values=([st.session_state.counter_two.index[:],st.session_state.counter_two["DeploymentID"][0:].tolist(),st.session_state.counter_two["clean_MasterUseCase"][0:].tolist()]))))
                                
                                            fig.update_layout(width=500,height=350,margin=dict(l=1,r=1,b=10,t=10),
                                            paper_bgcolor = background_color
                                            
                                            )
                                            st.write(fig)
                                    with col2:
                                        if st.session_state.radio=="Search In Master Usecase Table":
                                            
                                            placeholder_result = st.empty()
                                            result = placeholder_result.text_input('Row Number')
                                            st.session_state.counter_three=placeholder_result  
                                            btn=st.form_submit_button('Fetch')
                                            if btn:
                                                try:
                                                    
                                                        
                                                    if isinstance(int(result.strip()), int):
                                                        df=pd.DataFrame(st.session_state.counter_two.loc[int(result)]).T
                                                        
                                                        st.dataframe(df.style.set_properties(**{'background-color': '#FFA500',
                                                               'color': 'white'}))
                                                        #st.write(pd.DataFrame(st.session_state.counter_two.loc[int(result)]).T)
                                                    
                                                    
                                                except Exception as e:
                                                     st.error(e)
                                                     #placeholder_result.text_input('Row Number',value='')
                                                     placeholder_result.text_input('Row Number',value='')
                    except:
                     pass
        
        
            

if __name__=="__main__":

    main()
