import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
st.title("Netscore Analytics")

def date_clean(dataframe):
    dataframe['Date']=pd.to_datetime(dataframe['Date'])
    dataframe['Year']=dataframe['Date'].dt.year
    dataframe['Month']=dataframe['Date'].dt.month
    dataframe['Day']=dataframe['Date'].dt.day_name()
    dataframe['Quarter']=dataframe['Date'].dt.quarter
    return dataframe

def sfr(dataframe):
    for i in dataframe.select_dtypes(include='object').columns:
        dataframe[i]=dataframe[i].str.capitalize()
    return dataframe


def one_dim(dataframe,x,y,z):
    return dataframe[y].groupby(dataframe[x]).agg(z).reset_index()


def top_bottom(dataframe,value,order):
    if order=='top':
        return dataframe.sort_values(by=value,ascending=False).head(5)
    else:
        return dataframe.sort_values(by=value).head(5)

def two_dim(col_name,col_value,dataframe,year=None):
    if year==None:
        return dataframe.loc[(dataframe['Year']==None)|(dataframe[col_name]==col_value)]
    else:
        return dataframe.loc[(dataframe['Year']==year)&(dataframe[col_name]==col_value)]

new_data=pd.read_csv("C:\\Users\\ajay\\Downloads\\Jonathan.csv")
data=new_data.iloc[:,1:]
data=data.loc[data['Amount']>0]
data=sfr(date_clean(data))
filter=st.sidebar.selectbox("Column Filters",data.columns)
features=data.columns
n,c,g,r=st.columns([3,3,3,3])
with n:
    num_col = st.selectbox("Choose a column", options=data.select_dtypes(exclude=['object','datetime64[ns]','int64']).columns)
with c:
    c_val=st.selectbox("Choose a value",options=data[filter].unique())
with g:
    grpby=st.selectbox("Groupby Categories",options=data.select_dtypes(include=['object','int64'],exclude=['datetime64[ns]']).columns)
with r:
    reports=st.selectbox("Choose a Report",options=['Table','Graph','Shipments'])
if filter in features:
    if 'Table' in reports:
        col1,col2=st.columns([1,3])
        with col1:
            arithmetic=st.selectbox("Select a Metric",options=['sum','mean','count','max','min'])
        with col2:
            st.write("Total"+" "+arithmetic+" "+num_col+" "+"By"+filter+" "+"and"+" "+grpby)
            res=one_dim(two_dim(filter,c_val,data),grpby,num_col,arithmetic)
            st.table(res.head(25))
            st.download_button('Download the results',res.to_csv())
        if st.checkbox("Sort the values by order"):
            rad=st.radio("Sort the results",['Top 5','Bottom 5'])
            if "Top 5" in rad:
                st.table(top_bottom(one_dim(two_dim(filter, c_val, data), grpby, num_col,arithmetic), num_col, 'top'))
            else:
                st.write(top_bottom(one_dim(two_dim(filter, c_val, data), grpby, num_col, arithmetic), num_col, 'bottom'))

    elif "Graph" in reports:
        col3,col4=st.columns([2,2])
        with col3:
            chart=st.selectbox("Select below chart",options=['bar','pie','line','scatter'])
        with col4:
            arith = st.selectbox("Calculation", options=['sum', 'mean', 'min', 'max', 'count'])
        st.write("Total" + " " + arith + " " + num_col + " " + "By" + filter + " " + "and" + " " + grpby)
        result = one_dim(two_dim(filter, c_val, data), grpby, num_col, arith).sort_values(by=num_col,
                                                                                          ascending=False).head(10)
        st.write("Top 10 to display")
        if 'bar' in chart:
            X=result[result.columns[0]]
            Y=result[result.columns[1]]
            fig = go.Figure(data=[go.Bar(x=X,y=Y)])
            st.plotly_chart(fig, use_container_width=True)
        elif 'pie' in chart:
            labels=result[result.columns[0]]
            values=result[result.columns[1]]
            fig=go.Figure(data=[go.Pie(labels=labels,values=values,textinfo='label+percent')])
            st.plotly_chart(fig,use_container_width=True)
        elif 'line' in chart:
            X = result[result.columns[0]]
            Y = result[result.columns[1]]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X, y=Y,
                                     mode='lines+markers',
                                     name='lines'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            X = result[result.columns[0]]
            Y = result[result.columns[1]]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X, y=Y,
                                     mode='markers'))
            st.plotly_chart(fig, use_container_width=True)
    elif "Shipments" in reports:
        col5,col6=st.columns([3,3])
        with col5:
            yr=st.selectbox("Choose an Year",options=np.sort(data['Year'].value_counts().index))
        with col6:
            cate=st.selectbox("Choose a column",options=np.sort(data.select_dtypes(exclude=['datetime64[ns]','int64','float64']).columns))
        ord_res=pd.DataFrame([i for i in one_dim(data.loc[(data['Year']==yr) & (data[filter]==c_val)],grpby,cate,lambda x:x.describe()).set_index([grpby])[cate]],
            index=np.sort(data.loc[(data['Year']==yr) & (data[filter]==c_val),grpby].value_counts().index),
            columns=['Total','Unique','Highest','Frequent'])
        ord_res=ord_res.dropna()
        st.table(ord_res)
        ordr=st.select_slider("Select an order",options=['Top5','Bottom5'])
        if 'Top5' in ordr:
            st.write(top_bottom(ord_res,'Total','top'))
        else:
            st.write(top_bottom(ord_res,'Total','bottom'))








