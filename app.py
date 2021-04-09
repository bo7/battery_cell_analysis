
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
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_table
from dash.exceptions import PreventUpdate
from urllib.parse import quote as urlquote

from flask import Flask, send_from_directory

UPLOAD_DIRECTORY = os.path.dirname(os.path.realpath(__file__))+"/data/"
print(os.path.dirname(os.path.realpath(__file__)))
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


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
    db = UPLOAD_DIRECTORY+db
    print(db)
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


def create_statistics(ldft, min_max, sdev=4, customer = 'Generic'):
    lcols = [] # columns with UID
    batt_list = []
    kind_list = []
    value_list = [] 
    x = {}
    df_half = int(len(ldft)/2) # second half is outlier
    #for i in range(df_half, len(ldft)):
    #    lcols.extend(ldft[i].columns)
    x = { "Battery " + str(i%2): ldft[i].columns.to_list() for i in range(df_half, len(ldft)) }
    print(x)
    # lcols = list(set(lcols))
    # lcols.remove("mean")
    # lcols.remove("date_time")

    
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
    kind_list.append("Duration [min:s]:")
    value_list.append(str(int(minutes[0]))+":"+str(int(minutes[1])))
    #batt_list.append("-")
    kind_list.append("Cells per battery")
    value_list.append(date_list[4])
    #batt_list.append("-")
    kind_list.append("# Measurements")
    value_list.append(date_list[3])
    kind_list.append("Threshold sd")
    value_list.append(sdev)
    if len(x) == 0:
        kind_list.append("Anomalies")
        value_list.append(0)
    else:
      kind_list.append("Detected UIDs")
      value_list.append(" ")
      for key in x:
          for i in x[key][:-2]: # skip mean and date
            kind_list.append(key)
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
                        name=col_name,
                        line_width=.5 ))
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
            showlegend=True,
            legend_title="UID no",
            font=dict(
                family="Courier New, monospace",
                size=16,
                color="RebeccaPurple"
            ))
        fig.update_yaxes(showgrid=True, zeroline=False, showticklabels=True,
                 showspikes=True, spikemode='across', spikesnap='cursor', showline=True, spikedash='solid'
                 ,spikethickness=0.5)

        fig.update_xaxes(showgrid=True, zeroline=False, rangeslider_visible=False, showticklabels=True,
                 showspikes=True, spikemode='across', spikesnap='cursor', showline=True, spikedash='solid',spikethickness=0.5)

        fig.update_layout(hoverdistance=0)
        fig.update_traces(xaxis='x', hoverinfo='none')
        if j < len(df)/2:
            fig.update_layout(showlegend=False)
            children.append(dcc.Graph(id='graph-{}'.format(j),figure=fig))
        else:
            children.append(dcc.Graph( id={'type': 'graph','index': j},figure=fig))
        fig = go.Figure()
    return children

def create_stats_table(df_stats):
    res = []
    res.append(html.H6("Statistics"))
    res.append(dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in df_stats.columns],
    data=df_stats.to_dict('records'),
    fixed_rows={'headers': True},
    style_cell={
        'minWidth': 80, 'maxWidth': 80, 'width': 80, 'textAlign': 'left', 'backgroundColor': 'rgb(50, 50, 50)',
        'color': 'white',
    },
    
    style_as_list_view=True,
    ))
    #print(res)
    return res

# Initialise the app
server = Flask(__name__)
app = dash.Dash(server=server, suppress_callback_exceptions = True, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.config.suppress_callback_exceptions = True
###### globals ############
gsdev = 5
glists = []
gdf_stats = pd.DataFrame()
########################

form = dbc.Jumbotron(
    [
        html.H6("Export for Excel and PDF statistics"),
    dbc.Form(
        [ 
        dbc.FormGroup(
            [
                
                dbc.Input(type="text", placeholder="Enter customer", className="mr-3"),
                dbc.Button("Export", color="primary", className="mr-3"),
            ],
            
        ),
        ],
    inline=True,
), ])

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Cells", href="/page-1")),
        dbc.DropdownMenu(
            children=[
                
                dbc.DropdownMenuItem("Upload Databases", href="/page-2"),
            ],
            nav=True,
            in_navbar=True,
            label="More",
        ),
    ],
    brand="Sileo Cell Analysis",
    brand_href="/page-1",
    color="primary",
    dark=True,
)
###########################

# Define the app
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


page_1_layout = html.Div(children=[navbar,
                      html.Div(className='row',  # Define the row element
                               children=[
                                  html.Div(className='two columns div-user-controls'
                                  ,children = [
                                   # html.H2('Sileo Cell Analysis'),
                                      
                                     
                                    dcc.Upload(
                                        id="upload-data",
                                        children=html.Div(
                                           [dbc.Button("Select DB",  id="upload-button", outline=True, size ="lg", color="primary", className="mr-1")]
                                        ),
                                        
                                        multiple=True,
                                        ),
                                        html.Div(id='output-data-upload'),
                                       # html.Div(children = [
                                    # ], id = "statistics"),
                                ]),  # Define the left element
                                  html.Div(className='ten columns div-for-charts bg-white'
                                  , children =[
                                     html.Div(children =[

                                     ], id = "graphs"),
                                     html.Div(children =[
                                     ], id = "slider"),
                                     html.Div(children = [
                                           ], id = "form_customer"),
                                      html.Div(children = [
                                     ], id = "statistics"),
                                      
                                     
                                  ], id = "container")  # Define the right element
                                  ])
                                ])

page_2_layout = html.Div(children=[navbar,
                      html.Div(className='row',  # Define the row element
                               children=[
                                  html.Div(className='four columns div-user-controls'
                                  ,children = [
                                    html.H2('Sileo Cell Analysis'),
                            
                                     html.P(),
                                     html.Div(
    [
        html.H3("File Browser"),
        dcc.Upload(
            id="upload-data",
            children=html.Div(
                ["Drag and drop or click to select a file"]
            ),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=True,
        ),
    ],
    style={"max-width": "300px"},
)

                                ],),  # Define the left element
                                  html.Div(className='eight columns div-for-charts bg-white'
                                  , children =[
                                     html.Div(children =[
                                         html.H2("File List"),
                                         html.Ul(id="file-list"),
                                     ], id = "XXXX"),
                                  ], id = "container")  # Define the right element
                                  ])
                                ])

@app.callback(
              [Output('graphs', 'children'),
              Output('slider', 'children'),
              Output('statistics', 'children'),
            Output('form_customer', 'children'),],
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'),)
def update_output(content, name, date):
    children = []
    slider1 = []
    stats_table = []
    form_div = []
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
        df_stats = create_statistics(df, rows, sdev=gsdev, customer = 'Generic')
        global grows
        grows = rows
        global glists 
        glists = ldft
        #print(df[0].head())
        children = create_graphs(df)
        stats_table = create_stats_table(df_stats)
        slider1.append(html.P())
        slider1.append(dcc.Slider(
                            id = "slider_sd",
                            min=0,
                            max=10,
                            step= 1,
                            marks={i: 'SD {}'.format(i) for i in range(11)},
                            value=gsdev
                        ) )
        form_div.append(html.Br())
        form_div.append(form)
    #print(slider1)       
    if len(children) == 0:
        children.append(html.H5("")) 
        slider1.append(html.H5("")) 
        stats_table.append(html.H5("No Stats"))
        form_div.append("")
    return children , slider1 , stats_table, form_div

@app.callback([Output({'type': 'graph', 'index': ALL}, 'figure'),
               Output('table', 'data'),
               Output('table', 'columns')],
               Input('slider_sd', 'value'),)
def change_slider(value):
    ldft = []
    children = []
    stats_table =[]
    figs = []
    if value is None:
        return dash.no_update
    if value is not None:
        ldft = create_data_lists(glists,value)
        global gsdev
        gsdev = value
        df_stats = create_statistics(ldft, grows, sdev=gsdev, customer = 'Generic')
        children = create_graphs(ldft)
        stats_table = create_stats_table(df_stats)
        print(df_stats)
        #for count, fig in enumerate(children): # step through all graphs doesnt matter of battery count
        for count in range(int(len(children)/2),len(children)): # only graps > len/2 are sd amendments
            figs.append(children[count].figure) # add dynamic for all graphs to use ALL
        return figs,  stats_table[1].data, stats_table[1].columns
   # return fig1, fig2

@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout
    elif pathname == '/page-2':
        return page_2_layout
    else:
        return page_1_layout
    # You could also return a 404 "URL not found" page here


def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))


def uploaded_files():
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files


def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "/download/{}".format(urlquote(filename))
    return html.A(filename, href=location)


@app.callback(
    Output("file-list", "children"),
    [Input("upload-data", "filename"), Input("upload-data", "contents")],
)
def update_output_list(uploaded_filenames, uploaded_file_contents):
    """Save uploaded files and regenerate the file list."""

    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            save_file(name, data)

    files = uploaded_files()
    if len(files) == 0:
        return [html.Li("No files yet!")]
    else:
        return [html.Li(file_download_link(filename)) for filename in files]

if __name__ == '__main__':
    app.run_server(debug=True)

