#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from ipyleaflet import Map, Heatmap, Polygon, Circle
from ipywidgets import HTML
from random import uniform
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from pyproj import Proj, transform
import shapefile

from jupyter_dash import JupyterDash
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


# In[47]:


def load_data(path):
    df = pd.read_csv(path)
    h_df = df[df['SEMEL_YISHUV']==4000]

    inProj = Proj(init='epsg:2039')
    outProj = Proj(init='epsg:4326')

    def transform_coord(s):
        s['lon'], s['lat'] = transform(inProj, outProj, s['X'], s['Y'])
        return s

    print(h_df.shape)
    h_df = h_df.apply(transform_coord, axis=1)

    return h_df


parser = argparse.ArgumentParser()
parser.add_argument('-file')
input_path = parser.parse_args().file
# print('input_path', input_path)
    # H20161332Accdatamekuzar.csv
if input_path is not None:
    h_df = load_data(input_path)
    h_df_mask = (h_df['STATUS_IGUN']==1)
else:
    h_df = pd.read_csv('h_df.csv')
    h_df_mask = (h_df['SHNAT_TEUNA'] >= 2017) & (h_df['STATUS_IGUN']==1)



h_df = h_df[h_df_mask]
h_df.shape


# In[49]:


codes = pd.read_excel(r'Codebook.xlsx', header=4)
codes = codes.drop(['Unnamed: '+str(i) for i in range(5, 17)], errors='ignore', axis='columns')
codes.columns = ['col_meaning', 'col_name', 'value', 'value_meaning', 'explains']

codes = codes[codes['value'].notna()]
codes['col_name'] = codes['col_name'].fillna(method='ffill')
# codes['code'] = codes['col_name']+'_'+codes['value'].astype('str')
codes

m = codes.sort_values(['col_name', 'value']).set_index(['col_name', 'value'])['value_meaning'].to_dict()

codes_dict = {}
for k, v in m.items():
    if k[0] not in codes_dict:
        codes_dict[k[0]] = {}
    codes_dict[k[0]][k[1]] = v

codes_dict['RAV_MASLUL'][0.0] = 'לא קיים'
codes_dict['HAD_MASLUL'][0.0] = 'לא קיים'

streets = pd.read_csv(r'streets.csv', encoding='iso8859_8')
streets = streets[streets['שם_ישוב']=='חיפה ']
streets_dict = streets.set_index('סמל_רחוב')['שם_רחוב'].to_dict()
codes_dict['REHOV1'] = streets_dict
codes_dict['REHOV2'] = streets_dict

columns_dict = {'SHAA': 'שעה',
                'SUG_YOM': 'סוג יום',
                'YOM_LAYLA': 'יום \ לילה',
                'YOM_BASHAVUA': 'יום בשבוע',
                'HUMRAT_TEUNA': 'חומרת תאונה',
                'SUG_TEUNA': 'סוג תאונה',
                'HAD_MASLUL': 'חד מסלולית',
                'RAV_MASLUL': 'רב מסלולית',
                'MEHIRUT_MUTERET': 'מהירות מותרת',
                'TKINUT': 'תקינות הדרך',
                'ROHAV': 'רוחב הכביש',
                'SIMUN_TIMRUR': 'סימון ותמרורים',
                'TEURA': 'תאורה',
                'MEZEG_AVIR': 'מזג אוויר',
                'PNE_KVISH': 'פני הכביש',
                'date': 'תאריך',}

h_df = h_df.replace(codes_dict).rename(columns=columns_dict)


# In[50]:



shape = shapefile.Reader(os.path.join("StatZones", "Stat_Zones.shp"))
stat_zones_polygons = {r.record[1]: r.shape for r in shape.shapeRecords()}  # {stat_zone_code : shape}

# project Israel TM Grid to wgs84
proj = Proj("+proj=tmerc +lat_0=31.7343936111111 +lon_0=35.2045169444445 +k=1.0000067 +x_0=219529.584 +y_0=626907.39 +ellps=GRS80 +towgs84=-24.002400,-17.103200,-17.844400,-0.33077,-1.852690,1.669690,5.424800 +units=m +no_defs")

zones_coords = {stat_zone_code: [list(proj(x, y, inverse=True)) for (x, y) in polygon.points]
                for stat_zone_code, polygon in stat_zones_polygons.items()}
zones_polygons = {stat_zone_code: Polygon(coord_list)
                  for stat_zone_code, coord_list in zones_coords.items()}
zones_lonlats = {stat_zone_code: ([coords[0] for coords in coord_list], 
                                  [coords[1] for coords in coord_list])
                 for stat_zone_code, coord_list in zones_coords.items()}


# In[51]:


def in_hadar(r):
    isin = False
    for zone_polygon in zones_polygons.values():
        isin |= zone_polygon.contains(Point(r['lon'], r['lat']))
    return isin

h_df['in_hadar'] = h_df.apply(in_hadar, axis='columns')
h_df['תאריך'] = h_df.apply(lambda r: str(int(r['HODESH_TEUNA'])).zfill(2) + '/' + str(int(r['SHNAT_TEUNA'])), axis=1)

# KNN
X = h_df[['lat', 'lon']].values
X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

k = 8
neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(X_norm)
dists, _ = neigh.kneighbors(X_norm)
ddf = pd.DataFrame(dists)
# display(ddf.head())
s = ddf[1].sort_values().reset_index(drop=True)
plt.plot(s)
plt.grid()
# plt.ylim(0, 0.004)


# In[52]:


# heuristic value for eps param
ddf = pd.DataFrame(s)
ddf.columns = ['values']
ddf['diff'] = ddf['values'].diff()
ddf['pct'] = ddf['diff'].pct_change()
ddf = ddf[ddf['pct'].between(1, 9999999)].reset_index()

def strike(x):
    if ((x.diff().sum() <= 6)):
        return True
    else:
        return False

ddf['strike'] = ddf['index'].rolling(3).apply(strike)
ddf = ddf[ddf['strike']==1]
if ddf.empty:
    # default value
    eps = 0.0075
else:
    eps = ddf.iloc[0, ddf.columns.get_loc('values')]
    
print('eps', eps)

# heuristic value for min_samples param
min_samples = h_df['SHNAT_TEUNA'].nunique() * 2 + 1

print('min_samples', min_samples)

# In[53]:


X = h_df[['lat', 'lon']].values

# Compute DBSCAN
X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_norm)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

h_df['label'] = labels
h_df['noise'] = h_df['label']==-1
h_df['label_str'] = h_df['label'].astype(str)
h_df['size'] = h_df['label_str'].apply(lambda l: 1 if l != '-1' else 0.25)
h_df['pedestrians'] = h_df['סוג תאונה']=='פגיעה בהולך רגל'

ps_df = pd.DataFrame(h_df.sort_values('label')['label'].unique(), columns=['label'])
ps_df['label_str'] = ps_df['label'].astype(str)
ps_df['pseudo'] = True
ps_df['in_hadar'] = False
ps_df['pedestrians'] = False
ps_df['lon'] = 0
ps_df['lat'] = 0
ps_df['size'] = 0
h_df['pseudo'] = False

h_df = pd.concat([h_df, ps_df]).reset_index()
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Data size:', len(X))
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

all_hotspots = h_df.sort_values('label')['label_str'].unique()


# In[61]:


hotspots_names = {}
for k in all_hotspots:
    r1 = h_df[h_df['label_str']==k]['REHOV1'].mode()[0]
    try:
        r2 = h_df[h_df['label_str']==k]['REHOV2'].mode()[0]
    except:
        r2 = ''
    if r1 == r2:
        hotspots_names[k] = f'{k}: {r1}'
    else:
        name = '/'.join([r1, r2])
        hotspots_names[k] = f'{k}: {name}'
hotspots_names['-1'] = '1-: אחר'
hotspots_names['hotspots'] = 'hotspots'
hotspots_names[('hotspots', True)] = 'hotspots: הולכי רגל בלבד'
hotspots_names[('hotspots', False)] = 'hotspots: ללא הולכי רגל'
hotspots_names[('-1', True)] = '1-: אחר: הולכי רגל בלבד'
hotspots_names[('-1', False)] = '1-: אחר: ללא הולכי רגל'

h_df['label_name'] = h_df['label_str'].apply(lambda l: hotspots_names[l])


# In[62]:


wdf = h_df.copy()
wdf = wdf[~wdf['pseudo']]

# values that dont mean anything special
irrelevant_dummies = ['SUG_YOM_4.0', 'SUG_TEUNA_15.0', 'TKINUT_0.0', 'TKINUT_1.0', 
                      'SIMUN_TIMRUR_3.0', 'SIMUN_TIMRUR_4.0', 'SIMUN_TIMRUR_5.0', 
                      'ROHAV_0.0', 'HAD_MASLUL_0.0', 'HAD_MASLUL_4.0', 'HAD_MASLUL_9.0', 
                      'RAV_MASLUL_0.0', 'RAV_MASLUL_5.0', 'MEHIRUT_MUTERET_0.0', 
                      'MEZEG_AVIR_9.0', 'TEURA_1.0', 'TEURA_6.0', 'TEURA_3.0', 
                      'PNE_KVISH_9.0', 'PNE_KVISH_5.0', 'TEURA_11.0', 'MEZEG_AVIR_5.0']
irrelevant_dumm_cod = ['SUG_YOM_יום אחר', 'SUG_TEUNA_אחר', 'TKINUT_לא ידוע', 'TKINUT_אין ליקוי', 
                      'SIMUN_TIMRUR_אין ליקוי', 'SIMUN_TIMRUR_לא נדרש תמרור', 'SIMUN_TIMRUR_לא ידוע', 
                      'ROHAV_לא ידוע', 'HAD_MASLUL_לא קיים', 'HAD_MASLUL_אחר', 
                      'HAD_MASLUL_לא ידוע מספר מסלולים', 'RAV_MASLUL_לא קיים', 'RAV_MASLUL_אחר', 
                      'MEHIRUT_MUTERET_לא ידוע', 'MEZEG_AVIR_לא ידוע', 'TEURA_אור יום רגיל', 
                      'TEURA_לילה לא ידוע', 'TEURA_לילה פעלה תאורה', 'PNE_KVISH_לא ידוע', 
                      'PNE_KVISH_חול או חצץ על הכביש', 'TEURA_יום לא ידוע', 'MEZEG_AVIR_אחר']
dummy_columns = ['SHAA', 'SUG_YOM', 'YOM_BASHAVUA', 
                 'HUMRAT_TEUNA', 'SUG_TEUNA', 
                 'TKINUT', 'ROHAV', 'SIMUN_TIMRUR', 'TEURA', 'MEZEG_AVIR', 'PNE_KVISH', ]
multiplaces_columns = 'MEHIRUT_MUTERET', 'HAD_MASLUL', 'RAV_MASLUL'
relevant_columns = ['SUG_TEUNA', 'HUMRAT_TEUNA', 
                    'TKINUT', 'SIMUN_TIMRUR', 'ROHAV', 
                    'HAD_MASLUL', 'RAV_MASLUL', 'MEHIRUT_MUTERET']
one_rule_col = ['SHAA', 'YOM_BASHAVUA', 'HUMRAT_TEUNA', 'SUG_TEUNA', 'TKINUT', 
                'SIMUN_TIMRUR', 'TEURA', 'MEZEG_AVIR', 'PNE_KVISH',]
must_columns = ['SUG_TEUNA', 'HUMRAT_TEUNA']
dummy_columns = [columns_dict[c] for c in dummy_columns]
multiplaces_columns = [columns_dict[c] for c in multiplaces_columns]
relevant_columns = [columns_dict[c] for c in relevant_columns]
must_columns = [columns_dict[c] for c in must_columns]
one_rule_col = [columns_dict[c] for c in one_rule_col]
rules_dict = {}

for g_label, g_df in wdf.groupby('label_str'):
    if g_label == '-1':
        continue
    print(g_label, end=', ')
    old_cols = g_df.columns.tolist()
    
    g_df = g_df.replace(codes_dict)

    g_dummy = pd.get_dummies(g_df, columns=dummy_columns, prefix_sep=': ')
    
    g_dummy = g_dummy.drop(columns=set(irrelevant_dumm_cod+irrelevant_dummies+old_cols)
                           -set(dummy_columns), 
                           errors='ignore')
        
    one_rule = apriori(g_dummy, max_len=1, use_colnames=True, min_support=0.3)
    one_rule["itemsets"] = one_rule["itemsets"].apply(lambda x: list(x)[0])
    one_rule = one_rule[one_rule.apply(lambda r: (r['itemsets'].split(': ')[0]) in one_rule_col, axis=1)]
    one_rule = one_rule.sort_values('support', ascending=False).round(3)
    
    one_rule = one_rule.rename(columns={'itemsets': 'מופע'})
    one_rule.columns = [c.title() for c in one_rule.columns]

    rules_dict[g_label] = one_rule

print()
for g_label, g_df in wdf.groupby(['noise']):
    print(g_label, end=', ')
    old_cols = g_df.columns.tolist()

    dummy_columns += multiplaces_columns
    
    g_df = g_df.replace(codes_dict)

    g_dummy = pd.get_dummies(g_df, columns=dummy_columns, prefix_sep=': ')
    g_dummy = g_dummy.drop(columns=set(irrelevant_dummies+old_cols)-set(dummy_columns), errors='ignore')
    
    frequent_itemsets = apriori(g_dummy, max_len=2, use_colnames=True, min_support=0.05)

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0])
    rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0])

    relevant_mask = (rules.apply(lambda r: (r['antecedents'].split(': ')[0]) in relevant_columns, axis='columns') & 
                     rules.apply(lambda r: (r['consequents'].split(': ')[0]) in relevant_columns, axis='columns') )
    relevant_mask = relevant_mask &                     (rules.apply(lambda r: (r['antecedents'].split(': ')[0]) in must_columns, axis='columns') |  
                     rules.apply(lambda r: (r['consequents'].split(': ')[0]) in must_columns, axis='columns') )
    
    rules = rules[relevant_mask & (rules['confidence']>0.3)]
    rules = rules.sort_values('lift', ascending=False).round(3)
    
    rules = rules.rename(columns={'antecedents': 'מופע', 'consequents': 'השלכה'})
    rules.columns = [c.title() for c in rules.columns]
    
    rules_dict[{True: '-1', False: 'hotspots'}[g_label]] = rules

print()
for g_label, g_df in wdf.groupby(['noise', 'pedestrians']):
    print(g_label, end=', ')
    old_cols = g_df.columns.tolist()

    dummy_columns += multiplaces_columns
    
    g_df = g_df.replace(codes_dict)

    g_dummy = pd.get_dummies(g_df, columns=dummy_columns, prefix_sep=': ')
    
    g_dummy = g_dummy.drop(columns=set(irrelevant_dummies+old_cols)-set(dummy_columns), errors='ignore')
    
    frequent_itemsets = apriori(g_dummy, max_len=2, use_colnames=True, min_support=0.05)

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0])
    rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0])

    relevant_mask = (rules.apply(lambda r: (r['antecedents'].split(': ')[0]) in relevant_columns, axis='columns') & 
                     rules.apply(lambda r: (r['consequents'].split(': ')[0]) in relevant_columns, axis='columns') )
    relevant_mask= relevant_mask &                     (rules.apply(lambda r: (r['antecedents'].split(': ')[0]) in must_columns, axis='columns') |  
                     rules.apply(lambda r: (r['consequents'].split(': ')[0]) in must_columns, axis='columns') )
    
    rules = rules[relevant_mask & (rules['confidence']>0.3)]
    rules = rules.sort_values('lift', ascending=False).round(3)
    
    rules = rules.rename(columns={'antecedents': 'מופע', 'consequents': 'השלכה'})
    rules.columns = [c.title() for c in rules.columns]

    rules_dict[({True: '-1', False: 'hotspots'}[g_label[0]], g_label[1])] = rules


# In[63]:


def get_map(wdf):
    fig = px.scatter_mapbox(wdf.sort_values('label'), lat="lat", lon="lon", opacity=1, mapbox_style="open-street-map",
                            hover_data=["חומרת תאונה"], size='size', size_max=10, 
                            labels={'label_str': 'Hot Spot'}, zoom=11.5, center={'lat':32.807, 'lon': 35.014},
                            color="label_str", color_discrete_sequence=['#000000']+px.colors.qualitative.Light24+px.colors.qualitative.Dark24, height=750)
    fig.update_layout(autosize=False, margin={"r":0,"t":0,"l":0,"b":0})
#     fig.update_layout()
    fig.update_layout(legend=dict(x=1, y=.5))

    for lonlat in zones_lonlats.values():
        fig.add_trace(
                      go.Scattermapbox(fill="toself", opacity=0.5, showlegend=False,
                                       lon=lonlat[0], lat=lonlat[1], hoverinfo='none',
                                       marker={'size': 0, 'color': "orange"}, 
                                       fillcolor='rgba(0, 0, 255, 0.2)') )

    return fig

def get_data_table(fdf):
    fdf = fdf.sort_values(['SHNAT_TEUNA', 'HODESH_TEUNA'], ascending=False)
    display_cols = ['date', 'YOM_LAYLA', 'HUMRAT_TEUNA', 'SUG_TEUNA', 'TKINUT', 
     'SIMUN_TIMRUR', 'TEURA', 'MEZEG_AVIR', 'PNE_KVISH', 'MEHIRUT_MUTERET']
    display_cols.reverse()
    display_cols = [columns_dict[c] for c in display_cols]
    table_fig = go.Figure(data=[go.Table(
                                         header=dict(values=list(display_cols),
                                                     align='center'),
                                         cells=dict(values=[fdf[c] for c in display_cols],
                                                    align='right'),
                                         columnwidth=[9, 9, 9, 13, 9, 9, 15, 9, 9, 9]
                                        )
                                ]
                         )
    table_fig.update_layout(margin=dict(r=0, l=0, t=0, b=0))
    
    return table_fig
    

def get_rules_table(label):
    if (isinstance(label, tuple) and label[0] not in ['-1', 'hotspots']):
        label = label[0]
    rules_df = rules_dict[label]
    if label in ['-1', 'hotspots'] or        (isinstance(label, tuple) and label[0] in ['-1', 'hotspots']):
        display_cols = ['Support', 'Confidence', 'Lift', 'השלכה', 'מופע']

    else:
        display_cols = ['Support', 'מופע']

    rules_fig = go.Figure(data=[go.Table(
                                         header=dict(values=list(display_cols),
                                                     align='center'),
                                         cells=dict(values=[rules_df[c] for c in display_cols],
                                                    align='right'),
                                         columnwidth=[1, 1, 1, 2, 2],
                                        )
                                ]
                         )
    rules_fig.update_layout(margin=dict(r=0, l=0, t=0, b=0))
    return rules_fig


# In[64]:


# dashboard 
app = JupyterDash(__name__, external_stylesheets=external_stylesheets)
server= app.server

app.layout = html.Div([
    html.Table([
        html.Tr([
            html.Td(
                [html.Div([ dcc.Graph(figure=get_map(h_df) ,id='scattermap') ], 
                          style={'height':'100%','width': '100%'}, 
                          id='c11')
                ],
                style={'height':'60%','width': '50%'}),

            html.Td(
                [html.Div([ html.H6('חומרת תאונה'), 
                            dcc.Checklist(options=[{'label': 'קטלנית', 'value': 1},
                                                   {'label': 'קשה', 'value': 2},
                                                   {'label': 'קלה', 'value': 3}],
                                          value=[1, 2, 3],
                                          id='severity'),
                            html.H6('איזורי בחירה'), 
                            dcc.RadioItems(options=[{'label': 'הדר בלבד', 'value': 'hadar'},
                                                    {'label': 'כל חיפה', 'value': 'all'}],
                                          value='all',
                                          id='areas')],
                          style={'height':'100%','width': '100%'}, 
                          id='c12'),
                 html.Div([ html.H6('בחירת Hot Spots'), 
                            dcc.Checklist(id="all-hotspots",
                                          options=[{"label": "הכל", "value": "All"}],
                                          value=["All"],
                                          labelStyle={"display": "inline-block"}),
                            dcc.Checklist(options=[{'label': v, 'value': k}
                                                   for k, v in hotspots_names.items()
                                                   if isinstance(k, str) and k != 'hotspots'],
                                          value=all_hotspots,
                                          labelStyle={'border': '1px', 'transparent': 'solid', 
                                                 'display':'inline-block', 'width': '15em'},
                                          id='hotspots'),
                          ]),
                 html.Div([ html.H6('סוג תאונה'), 
                            dcc.Checklist(options=[{'label': 'פגיעה בהולכי רגל', 'value': 'pedestrians'},
                                                   {'label': 'אחר', 'value': 'others'}],
                                          value=['pedestrians', 'others'],
                                          labelStyle={'border': '1px', 'transparent': 'solid', 
                                                 'display':'inline-block', 'width': '12em'},
                                          id='acc-type'),
                          ],
                          style={'height':'60%','width': '30%'})
                ], 
                          style={'height':'60%','width': '30%', 'direction': 'rtl', 'text-align': 'right'})]),
        html.Tr([
            html.Td([html.Div([ html.H6('פרטי התאונות במפה (', id='data-title', style={'display': 'inline'}),
                                html.H6(f'{len(h_df)})', id='data-len', style={'display': 'inline'}),
                                dcc.Graph(figure=get_data_table(h_df), id='data-table') ],
                              id='c31',
                              style={'height': '100%','width': '100%', 'background-color':'hsl(0, 0%, 60%)','text-align': 'center'})],  
                    style={'height':'30%','width': '57%'}),
            html.Td([html.Div([ html.H6('חוקים - Hot Spots', id='rules-title'),
                                dcc.Graph(figure=get_rules_table('hotspots'), id='rules-table') 
                              ],
                              id='c33',
                              style={'height':'100%','width': '100%', 'background-color':'hsl(70, 50%, 80%)','text-align': 'center'})],  
                    style={'height':'30%','width': '43%'}, )
                ]),
        
    ])
])

import numpy as np


@app.callback(
    [Output('rules-title', 'children'),
     Output('rules-table', 'figure'),
     Output('data-table', 'figure'),
     Output('scattermap', 'figure'),
     Output('data-len', 'children')],
    [Input('severity', 'value'),
     Input('areas', 'value'),
     Input('hotspots', 'value'),
     Input('acc-type', 'value')])
def legend_select(severity, areas, hotspots, acc_type):    
    
    if len(hotspots) == 1:
        rules_key = str(hotspots[0])
    else:
        rules_key = 'hotspots'

    fdf = h_df.copy()
    print('0', fdf.shape)
    severity = [codes_dict['HUMRAT_TEUNA'][v] for v in severity]
    fdf = fdf[(fdf['חומרת תאונה'].isin(severity)) | (fdf['pseudo'])]
    print('1', fdf.shape)
    if areas == 'hadar':
        fdf = fdf[(fdf['in_hadar']) | (fdf['pseudo'])]
    print('2', fdf.shape)
    fdf = fdf[(fdf['label_str'].isin(hotspots)) | (fdf['pseudo'])]
    print('3', fdf.shape)
    if acc_type == ['pedestrians',]:
        fdf = fdf[(fdf['pedestrians']) | (fdf['pseudo'])]
        rules_key = (rules_key, True)
    elif acc_type == ['others',]:
        fdf = fdf[(~fdf['pedestrians']) | (fdf['pseudo'])]
        rules_key = (rules_key, False)
    elif acc_type == []:
        fdf = fdf[(fdf['pedestrians'].isin([])) | (fdf['pseudo'])]
    print('4', fdf.shape)
        
    fdf = fdf.sort_values('label')
        
    # if specific hotspot with only\out pedestrians
    if (isinstance(rules_key, tuple) and rules_key[0] not in ['-1', 'hotspots']):
        rules_key = rules_key[0]

    rules_table = get_rules_table(rules_key)
    
    map_fig = get_map(fdf)
    
    fdf = fdf[~fdf['pseudo']]
    data_table = get_data_table(fdf)
    table_size = f'{len(fdf)})'
    print('table_size', table_size)
    
    return [f'חוקים - {hotspots_names[rules_key]}', rules_table, data_table, map_fig, table_size]
    

@app.callback(
    Output("hotspots", "value"),
    Output("all-hotspots", "value"),
    Input("hotspots", "value"),
    Input("all-hotspots", "value"),
)
def sync_checklists(hotspots_selected, all_selected):
    ctx = dash.callback_context
    input_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if input_id == "hotspots":
        all_selected = ["All"] if set(hotspots_selected) == set(all_hotspots) else []
    else:
        hotspots_selected = all_hotspots if all_selected else []
    return hotspots_selected, all_selected

app.run_server(mode="external",port=8060)


# In[ ]:




