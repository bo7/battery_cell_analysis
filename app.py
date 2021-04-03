
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
import sqlite3
import numpy as np
import itertools
from datetime import datetime
import argparse
import os
import sys
import argparse
import base64
import io
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash_table
from dash.exceptions import PreventUpdate


def create_graph(df,fn="Zelldiagramm", customer="Generic", show=True, sdev = 4, df_statistics = [], excel=False, dir="./daten/", thick = 1):
    fig, axs = plt.subplots(len(df)-1,2, figsize=(16, 10), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .4, wspace=.1)
    axs = axs.ravel()
    n = -1 # skip last row with count for x-axis
    half = (len(df))/2 # determines what is battery data first half, outlier second half preparing subplot title
    for i in range(len(df)):
        for k in df[i].columns[:n]: # n shows if last column must be skipped or not, due to preparedf dataframes or anomalie
            axs[i].plot(df[i][k].index,df[i][str(k)], label = str(k), linewidth= thick )
        if i < half:
            axs[i].set_title("battery " +str(i))
            if excel:
                df[i].to_excel(dir +customer+"_battery_" +str(i)+ ".xlsx")
        else:
            axs[i].set_title("battery " +str(i % 2) + " " + "anomalie with sd = " +str(sdev))
            if excel:
                df[i].to_excel(dir + customer+"_battery_" +str(i % 2)+ "_anomalie_with_sd_" +str(sdev)+ ".xlsx")
        if len(df[i].columns) < 30:
            axs[i].legend(loc="upper right", title="Cell(s) to examine ", bbox_to_anchor=(1, 1), fontsize = 5)
    cell_text = []
    for row in range(len(df_statistics)):
        cell_text.append(df_statistics.iloc[row])
    column_labels = df_statistics.columns
    axs[i+1].axis('tight')
    axs[i+1].axis('off')
    axs[i+1].table(cellText=cell_text,colLabels=column_labels,loc="center",cellLoc ='left', colLoc='left')
    if i % 2 == 1:
        fig.delaxes(axs[i+2]) # remove empty drawing if printet figures are uneven
    plt.suptitle(customer,fontsize=20)
    plt.savefig(fn+".pdf")
    if show:
        plt.show()
    if excel:
        df_statistics.to_excel(dir + customer + "_stats.xlsx", index=False)


def prepare_df(df):
    df["date_time"] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = (df.reset_index().pivot(index='date_time', columns='UID', values='U_V'))
    df.reset_index(drop=True, inplace=True)
    df['mean'] = df.iloc[:, 0:len(df.columns)].mean(axis=1) 
    df['date_time'] = df.index
    df.columns = df.columns.astype(str)
    #print(len(df.columns))
    return df

def read_db(db):
    #con = sqlite3.connect(db)
    j = 0 # counter for battery numbers
    ldf = []
    try:
        con = sqlite3.connect(db)
    except sqlite3.Error as e:
        print(e)
    cur = con.cursor()
    cur.execute("SELECT distinct battery FROM scl_data")
    rows = cur.fetchall()  
    print("sql1")
    cur.execute("""
        SELECT 
            Battery,
            min(rt.date || " " || rt.time),
            max(rt.date || " " || rt.time),
            count( *),
            count( distinct Addr)         
        FROM   record_time rt
        INNER JOIN scl_data sd 
                    ON rt.record = sd.record 
        GROUP BY sd.Battery	   
    """)
    rows2 =cur.fetchall()
    print("sql2")
    for row in rows:
        df = pd.read_sql_query("""SELECT rt.date, 
        rt.time, 
        sd.battery, 
        sd.addr, 
        sd.u_v, 
        su.uid
        FROM   record_time rt 
        INNER JOIN scl_data sd 
                ON rt.record = sd.record 
        INNER JOIN scl_uid su 
                ON su.battery = sd.battery 
                    AND su.addr = sd.addr 
        WHERE  sd.u_v > 0 and su.battery = """ +str(j) ,   con)
        j += 1
        ldf.append(df)
    print("sql3")
    return ldf,rows2 #rows2 = min max datetime per battery

def find_outliers(dfT,sdev=4):
    res = []
    for i in dfT.columns:
        ana = find_anomalies(dfT[i],sdev)
        if len(ana)>0:
            for j in ana:
                res.append(dfT.index[dfT[str(i)] == j].tolist())   # check index -> should be UID
    return res 


def find_anomalies(data, sdev=4):
    #define a list to accumlate anomalies
    anomalies = []
    
    # Set upper and lower limit to 3 standard deviation
    random_data_std = np.std(data,ddof = 1) # 
    random_data_mean = np.mean(data)
    anomaly_cut_off = random_data_std * sdev 
    
    lower_limit  = random_data_mean - anomaly_cut_off 
    upper_limit = random_data_mean + anomaly_cut_off
    # Generate outliers
    for outlier in data:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
    return anomalies

def ret_anomalies_as_list_index(df,sdev=4):
    dfT = df.iloc[:,0:-2] # copy
    dfT = dfT.T
    dfT['mean'] = dfT.mean(axis=1)
    dfT.columns = dfT.columns.astype(str)
    dfret = find_outliers(dfT,sdev) 
    return dfret

def create_data_lists(ldf,sdev):
    ldft = []
    for j in range(len(ldf)):
        ldft.append(prepare_df(ldf[j]))
    #ldft[0].to_excel("create_data")
    for j in range(len(ldf)):
        test = ret_anomalies_as_list_index(ldft[j],sdev)
        #print(test)
        test = list(set(list(itertools.chain(*test)))) 
        test.append('mean') # 
        test.append('date_time')
        ldft.append(ldft[j][test]) # filter dataframe with list
    return ldft

def create_statistics(ldft,ldf, min_max, sdev=4, customer = 'Generic'):
    lcols = [] # columns with UID
    batt_list = []
    kind_list = []
    value_list = [] 

    df_half = int(len(ldft)/2) # second half is outlier
    for i in range(df_half, len(ldft)):
        lcols.extend(ldft[i].columns)

    lcols = list(set(lcols))
    lcols.remove("mean")
    lcols.remove("date_time")

    
    date_list = [item for t in min_max for item in t]
    
    kind_list.append("Customer")
    value_list.append(customer)
    kind_list.append("Creation date")
    value_list.append(date_list[1][:10])
    batt_list.append("-")
    kind_list.append("Finish date")
    value_list.append(date_list[2][:10])
    batt_list.append("-")
    kind_list.append("Start time")
    value_list.append(date_list[1][11:])
   # batt_list.append("-")
    kind_list.append("Stop time")
    value_list.append(date_list[2][11:])
    dt_start = datetime. strptime(date_list[1], '%d.%m.%Y %H:%M:%S')
    dt_end = datetime. strptime(date_list[2], '%d.%m.%Y %H:%M:%S')
    dt_diff = dt_end - dt_start  
    minutes = divmod(dt_diff.total_seconds(), 60) 
    #batt_list.append("-")
    kind_list.append("Duration minutes:")
    value_list.append(str(minutes[0]))
    kind_list.append("Duration seconds:")
    value_list.append(str(minutes[1]))
    #batt_list.append("-")
    kind_list.append("Cells per battery")
    value_list.append(date_list[4])
    #batt_list.append("-")
    kind_list.append("# Measurements")
    value_list.append(date_list[3])
    kind_list.append("Threshold anomalie detection with sd")
    value_list.append(sdev)
    if len(lcols) == 0:
        kind_list.append("Anomalies")
        value_list.append(0)
    else:
        for i in sorted(lcols):
            kind_list.append("Detected UID")
            value_list.append(i)

    df_statistics = pd.DataFrame(list(zip(kind_list, value_list)), 
               columns =[ 'Name', "Value"]) 
    
    return df_statistics

def create_data_dir(directory="./data"):
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_graphs(df):
    fig = go.Figure()
    children = []
   # print(len(df))
    for j in range(len(df)):
        #print(j)
        for k in df[j].columns[:-1]:
            col_name = str(k)
            fig.add_trace(go.Scatter(x=df[j].index, y=df[j][col_name],
                        mode='lines', # 'lines' or 'markers'
                        name=col_name))
        if j < len(df)/2: # if not flag only sd df are passed, only update on anomalie title
            title = "Battery " + str(j)
        else:
            title = "Battery " + str(j % 2) + " anomalies with sd= " + str(gsdev)
            
        fig.update_layout(
            title={'text' : title,
                    'y':0.9,
                    'x':0.4,
                    'xanchor': 'center',
                    'yanchor': 'top'}, 
            xaxis_title="Measurements count",
            yaxis_title="Voltage",
            legend_title="UID no",
            font=dict(
                family="Courier New, monospace",
                size=16,
                color="RebeccaPurple"
            ))
        children.append(dcc.Graph(id='graph-{}'.format(j),figure=fig))
        fig = go.Figure()
    return children

# Initialise the app
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True
###### globals ############
gsdev = 5
glists = []
###########################

# Define the app
app.layout = html.Div(children=[
                      html.Div(className='row',  # Define the row element
                               children=[
                                  html.Div(className='four columns div-user-controls'
                                  ,children = [
                                    html.H2('Sileo cell analysis'),
                            
                                     html.P()
                                    , dcc.Upload(
                                        id="upload-data",
                                        children=html.Div(
                                           [dbc.Button("Select DB", outline=True, size ="lg", color="primary", className="mr-1")]
                                        ),
                                        
                                        multiple=True,
                                        )
                                        ,html.Div(id='output-data-upload')
                                ]),  # Define the left element
                                  html.Div(className='eight columns div-for-charts bg-white'
                                  , children =[
                                     html.Div(children =[

                                     ], id = "graphs"),
                                     html.Div(children =[
                                     ], id = "slider"),
                                  ], id = "container")  # Define the right element
                                  ])
                                ])
@app.callback(
              [Output('graphs', 'children'),
              Output('slider', 'children')],
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(content, name, date):
    children = []
    slider1 = []
    if content is not None:
        db_string = ""
        db_string = name[0]
        try:
            if 'db' in db_string:
                ldft, rows = read_db(db_string)
                #ldft[0]["index"] = ldft[0].index
        except Exception as e:
            print(e)

        df = create_data_lists(ldft,gsdev)
        global glists 
        glists = ldft
        #print(df[0].head())
        children = create_graphs(df)
        slider1.append(html.P())
        slider1.append(dcc.Slider(
                            id = "slider_sd",
                            min=0,
                            max=10,
                            step= 1,
                            marks={i: 'SD {}'.format(i) for i in range(11)},
                            value=gsdev
                        ) )
    #print(slider1)       
    if len(children) == 0:
        children.append(html.H5("select db")) 
        slider1.append(html.H5("No slider")) 
    return children , slider1  

@app.callback([Output('graph-2', 'figure'),
               Output('graph-3', 'figure')],
               Input('slider_sd', 'value'))
def change_slider(value):
    ldft = []
    # fig1 = go.Figure()
    # fig2 = go.Figure()
    children = []
    if value is None:
        return dash.no_update
    if value is not None:
        print(len(glists))
        ldft = create_data_lists(glists,value)
        global gsdev
        gsdev = value
        children = create_graphs(ldft)
        return children[2].figure, children[3].figure
   # return fig1, fig2


if __name__ == '__main__':
    app.run_server(debug=True)

