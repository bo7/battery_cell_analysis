
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
from subprocess import call
import dash_table
from dash.exceptions import PreventUpdate
from urllib.parse import quote as urlquote
from zipfile import ZipFile
from flask import Flask, send_from_directory
from os.path import basename
import glob
import shutil
from os import walk

UPLOAD_DIRECTORY = os.path.dirname(os.path.realpath(__file__))+"/data/"
WORKING_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
#print(os.path.dirname(os.path.realpath(__file__)))
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

_, _, filenames2 = next(walk("./data"))
 

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
    ldbfbox = []
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
    for row in rows: # iterate over batteries
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
        WHERE  sd.u_v > 0 and su.battery = """ +str(j) + """ order by 1,2 """ ,   con)
        
        dfbox = pd.read_sql_query("""
        SELECT 
        rt.date, 
        rt.time, 
        rt.date 
       || "_" 
       || rt.time ascDate, 
       su.uid, 
       sd.u_v 
FROM   record_time rt 
       INNER JOIN scl_data sd 
               ON rt.record = sd.record 
       INNER JOIN scl_uid su 
               ON su.battery = sd.battery 
                  AND su.addr = sd.addr 
WHERE  sd.u_v > 0 
       AND su.battery = """ +str(j) + """ 
       AND rt.date 
           || "_" 
           || rt.time IN (SELECT rt.date 
                                 || "_" 
                                 || rt.time 
                          FROM   record_time rt 
                                 INNER JOIN scl_data sd 
                                         ON rt.record = sd.record 
                                 INNER JOIN scl_uid su 
                                         ON su.battery = sd.battery 
                                            AND su.addr = sd.addr 
                          WHERE  sd.u_v > 0 
                                 AND su.battery = """ + str(j) + """  
                          GROUP  BY rt.date, 
                                    rt.time 
                          ORDER  BY Max(sd.u_v) - Min(sd.u_v)DESC 
                          LIMIT  1); """, con)
        j += 1
        ldf.append(df)
        ldbfbox.append(dfbox)
    print("sql3")
    return ldf,rows2, ldbfbox #rows2 = min max datetime per battery

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
    ldfuid = []
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
        ldfuid.append([ldf[j].UID])
    return ldft, ldfuid


def create_statistics(ldft, min_max, sdev=4, customer = 'Generic'):
    #lcols = [] # columns with UID
    batt_list = []
    kind_list = []
    value_list = [] 
    x = {}
    df_half = int(len(ldft)/2) # second half is outlier
    #for i in range(df_half, len(ldft)):
    #    lcols.extend(ldft[i].columns)
    x = { "Battery " + str(i% df_half): ldft[i].columns.to_list() for i in range(df_half, len(ldft)) }
    #print(x)
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

def create_graphs(df,dfuid):
    fig = go.Figure()
    children = []
    #print(len(df))
    for j in range(len(df)):
        #print(j)
        for k in df[j].columns[:-1]:
            col_name = str(k)
            fig.add_trace(go.Scatter(x=df[j].index, y=df[j][col_name],
                        mode='lines', # 'lines' or 'markers'
                        name=col_name,
                        line_width=.5, ))
        if j < len(df)/2: # if not flag only sd df are passed, only update on anomalie title
            title = "Battery " + str(j)
        else:
            title = "Battery " + str(j % int(len(df)/2)) + " anomalies with sd= " + str(gsdev)
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
        fig.update_yaxes(showgrid=False , zeroline=False, showticklabels=True,
                 showspikes=True, spikemode='across', spikesnap='cursor', showline=True, spikedash='solid',spikethickness=0.5)

        fig.update_xaxes(showgrid=False, zeroline=False, rangeslider_visible=False, showticklabels=True,
                 showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid',spikethickness=0.5)

        
        fig.update_traces(xaxis='x' ,hoverinfo='text')
        fig.update_layout(hoverdistance=0, hovermode  = 'x')
        if j < len(df)/2:
            fig.update_layout(showlegend=False)
            fig.update_traces(text = dfuid[j], hovertext=dfuid[j])
            children.append(dcc.Graph(id='graph-{}'.format(j),figure=fig))
        else:
            children.append(dcc.Graph( id={'type': 'graph','index': j},figure=fig))
        fig = go.Figure()
    return children

def create_boxplot(ldfbox):
    box_res = []
    helpdf = pd.DataFrame()
    fig = go.Figure()
    fig2 = go.Figure()
    cnt_batteries = len(ldfbox)
    box_name = ""
    box_text = ""
    #battery 0 is plottet for biggest delta on voltage for timestamp battery 1 is compared same timestamp and vice versa
    if cnt_batteries == 2:
        for i in range(len(ldfbox)):
            if i == 0:
                helpdf['U_V'] = gsql[i+1][(gsql[i+1]['Date'] == ldfbox[i]['Date'][0]) & (gsql[i+1]['Time'] == ldfbox[0]['Time'][0]) ]['U_V']
                box_name = "Battery " + str(i+1) + " " +ldfbox[i]['ascDate'][0] 
                box_text =ldfbox[i+1]['UID']
            else:
                helpdf['U_V'] = gsql[i-1][(gsql[i-1]['Date'] == ldfbox[i]['Date'][0]) & (gsql[i-1]['Time'] == ldfbox[0]['Time'][0]) ]['U_V']
                box_name = "Battery " + str(i-1) + " " +ldfbox[i]['ascDate'][0] 
                box_text =ldfbox[i-1]['UID']
            fig.add_trace(go.Box(y=ldfbox[i]['U_V'] ,name = "Battery " + str(i) + " " +ldfbox[i]['ascDate'][0] ,text=ldfbox[i]['UID']))
            fig.add_trace(go.Box(y=helpdf['U_V'], name = box_name, text = box_text))
            fig2.add_trace(go.Histogram(x=ldfbox[i]['U_V']))#,name = "Battery " + str(i),text=ldfbox[i]['UID']))
            fig2.add_trace(go.Histogram(x=helpdf['U_V']))#['U_V'],name = "Battery " + str(i),text=ldfbox[i]['UID']))
        fig.update_traces( boxpoints='all', # can also be outliers, or suspectedoutliers, or False
            jitter=0.3, # add some jitter for a better separation between points
            pointpos=-1.8, # relative position of points wrt box          
            )
    fig.update_layout(
    title="Boxplot "
    )
    fig2.update_layout(
    title="Histogram "
    )
    # Overlay both histograms
    fig2.update_layout(barmode='stack')
    # Reduce opacity to see both histograms
    fig2.update_traces(opacity=0.75)
    box_res.append(dcc.Graph(id = 'div_box', figure = fig))
    box_res.append(dcc.Graph(id = 'div_hist', figure = fig2))
    return box_res
 

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
def zipFilesInDir(dirName, zipFileName, filter):
   # create a ZipFile object
   with ZipFile(zipFileName, 'w') as zipObj:
       # Iterate over all the files in directory
       for folderName, subfolders, filenames in os.walk(dirName):
           for filename in filenames:
               if filter(filename):
                   # create complete filepath of file in directory
                   filePath = os.path.join(folderName, filename)
                   # Add file to zip
                   zipObj.write(filePath, basename(filePath))

# Initialise the app
server = Flask(__name__)
app = dash.Dash(server=server, suppress_callback_exceptions = True, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.config.suppress_callback_exceptions = True

@server.route("/data/<path:path>")
def download(path):
    """Serve a file from the upload directory."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)

###### globals ############
gsdev = 5
glists = []
gdf_stats = pd.DataFrame()
gdbname = ""
gsql = [] # sql data from select 
########################

form = dbc.Jumbotron(
    [
        html.H6("Export for Excel and PDF statistics"),
    dbc.Form(
        [ 
        dbc.FormGroup(
            [
                
                dbc.Input(id="in_customer", type="text", placeholder="Enter customer", className="mr-3"),
                dbc.Button("Export", id ="export_button", color="primary", className="mr-3"),
                
            ],
            
        ),
        
        ],
    inline=True,
    
    ),html.Div(children=[dcc.Loading(
                    id="loading-1",
                    type="default",
                    children=html.Div(id="loading-output-1")
                ),], id='spinner'), 
     ])

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Cells", href="/page-1")),
        dbc.DropdownMenu(
            children=[
                
                dbc.DropdownMenuItem("Upload DB/Download Export ", href="/page-2"),
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
                                      
                                     
                                    dcc.Dropdown(
                                            id='file-dropdown',
                                            options=[
                                                {'label':name[-8:], 'value':name} for name in filenames2
                                            ],
                                           
                                            style = dict(
                                            width = '100%',
                                            display = 'inline-block',
                                            verticalAlign = "middle",
                                            backgroundColor ='green',
                                            color = 'green'
                                            ),
                                        ),
                                        html.Div(id='output-data-upload'),
                                       # html.Div(children = [
                                    # ], id = "statistics"),
                                ]),  # Define the left element
                                  html.Div(className='ten columns div-for-charts bg-white'
                                  , children =[
                                      html.Div(children=[dcc.Loading(
                                            id="loading-2",
                                            type="default",
                                            children=html.Div(id="loading-output-2")
                                        ),], id='spinner'), 
                                     html.Div(children =[

                                     ], id = "graphs"),
                                     html.Div(children =[
                                     ], id = "slider"),
                                     html.Div(children = [
                                           ], id = "boxplot"),
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
               Output('boxplot', 'children'),
              Output('statistics', 'children'),
              Output('form_customer', 'children'),
              Output('loading-output-2', 'children'),
              Output('file-dropdown', 'options')],
              dash.dependencies.Input('file-dropdown', 'value'),
              dash.dependencies.State("file-dropdown", "value"))
def update_output(name2, name):
    children = []
    slider1 = []
    stats_table = []
    form_div = []
    box_div = []
    global filenames2
    #print( filenames2) 
    print(name)
    #filenames2 = next(walk("./data"))
    _, _, filenames3 = next(walk("./data"))

    if name is not None:
        db_string = ""
        db_string = name #[0]
        try:
            if 'db' in db_string:
                global gdbname
                gdbname = db_string
                ldft, rows, ldfbox = read_db(db_string)
                global gsql
                gsql = ldft
                #ldft[0]["index"] = ldft[0].index
        except Exception as e:
            print(e)

        df, dfuid = create_data_lists(ldft,gsdev)
        df_stats = create_statistics(df, rows, sdev=gsdev, customer = 'Generic')
        #print(df)
        global grows
        grows = rows
        global glists 
        glists = ldft
        #print(df[0].head())
        children = create_graphs(df,dfuid)
        box_div = create_boxplot(ldfbox)
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
        box_div.append("")
    return children , slider1 , box_div, stats_table, form_div, html.H6(''), [{'label': i, 'value': i} for i in filenames3]

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
        ldft, dfuid = create_data_lists(glists,value)
        global gsdev
        gsdev = value
        df_stats = create_statistics(ldft, grows, sdev=gsdev, customer = 'Generic')
        children = create_graphs(ldft,dfuid)
        stats_table = create_stats_table(df_stats)
        #print(df_stats)
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
    location = "./data/{}".format(urlquote(filename))
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
#create_graph(df=ldft,fn=pdf_name, customer=customer, show=show_output, sdev = sdev, df_statistics = df_statistics, excel=excel_output, dir=output_dir, thick = thickness)

@app.callback(
    Output('loading-output-1', 'children'),
    [Input('export_button', 'n_clicks'),],
    State("in_customer", 'value')
)
def update_export_div(n_clicks, input_value):
    if os.path.exists("./tmp/.DS_Store"):
        os.remove("./tmp/.DS_Store")
    if os.path.exists("./data/.DS_Store"):
        os.remove("./data/.DS_Store")
    if not n_clicks:
        raise PreventUpdate
    if input_value is not None:
        #customer = '"'+ input_value + '"'
        customer = input_value.replace(" ", "_")# r'%s' % input_value
        pdf = '"Zelldiagramm ' + input_value + '"'#+ " "+ str(gdf_stats.iloc[1]["Value"])
        try:
            print(gdbname)
            db_name = '"' + gdbname + '"'
            print(('python3 cell_detection_0.91.py data/'  + db_name + ' -e -c ' + customer +' -s ' + str(gsdev) + ' -p ' +pdf))
            res =os.system('python3 cell_detection_0.91.py data/'  + db_name + ' -e -c ' + customer +' -s ' + str(gsdev) + ' -p ' +pdf)
            print(res)
        except Exception as e:
            print(e)
        zipFilesInDir('./tmp', customer +'.zip', lambda name : 'xlsx' or 'pdf' in name)
        files = glob.glob('./tmp/*')
        for f in files: 
            os.remove(f)
        shutil.move('./' +customer +'.zip', './data/')
        
        return html.H6(customer + ' exported')
    return html.H3(" ")

if __name__ == '__main__':
    app.run_server(debug=True)

