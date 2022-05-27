# -*- coding: utf-8 -*-
"""
Created on Tue May 24 14:25:22 2022

@author: SV
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import PySimpleGUI as sg
from matplotlib.widgets  import RectangleSelector
import matplotlib.figure as figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

fig = figure.Figure()
ax = fig.add_subplot(111)
DPI = fig.get_dpi()
fig.set_size_inches(505 * 2 / float(DPI), 707 / float(DPI))
############################# Graph Display ###################################
def draw_figure_w_toolbar(canvas, fig, canvas_toolbar):
    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()
    if canvas_toolbar.children:
        for child in canvas_toolbar.winfo_children():
            child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
    figure_canvas_agg.draw()
    toolbar = Toolbar(figure_canvas_agg, canvas_toolbar)
    toolbar.update()
    figure_canvas_agg.get_tk_widget().pack(side='right', fill='both', expand=1)


def line_select_callback(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    rect = plt.Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2) )
    print(rect)
    ax.add_patch(rect)


class Toolbar(NavigationToolbar2Tk):
    def __init__(self, *args, **kwargs):
        super(Toolbar, self).__init__(*args, **kwargs)
    

#################### Window Design ###########################   
sg.theme('BluePurple')
   
layout = [#[sg.Input(key='-IN-')],
          [sg.Button('Profit preconizat'), sg.Button('Exit'),sg.Button('Rata profit')],
          [sg.Text('Profitul preconizat/Rata profitului este:')],
          [sg.Text(size=(15,1), key='-OUTPUT-'),sg.Text(size=(15,1), key='-OUTPUT-')],
          [sg.B('Graph', key='Graph')],
    [sg.Canvas(key='controls_cv')],
    [sg.Column(
        layout=[
            [sg.Canvas(key='fig_cv',
                       # it's important that you set this size
                       size=(500 * 2, 700)
                       )]
        ],
        background_color='#DAE0E6',
        pad=(0, 0)
    )]]
  
window = sg.Window('DeepStart', layout,)
#################### Window Design ###########################


########################## PCA #####################
companies = pd.read_csv('1000_Companies2.csv')
companie = pd.read_csv('1000_Companies1.csv')
companies.head()
companie.head()
companies[['R&D Spend','Administration','Marketing Spend']].corr()
companie[['R&D Spend','Administration','Marketing Spend']].corr()
companies.drop(['Marketing Spend'],axis=1,inplace=True)
companie.drop(['Marketing Spend'],axis=1,inplace=True)


companies.drop(['State'],axis=1,inplace=True)
companie.drop(['State'],axis=1,inplace=True)

from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
companies[['R&D Spend','Administration']]=scale.fit_transform(companies[['R&D Spend','Administration']])
companie[['R&D Spend','Administration']]=scale.fit_transform(companie[['R&D Spend','Administration']])
y=companies.iloc[:,2].values 
y_T=companie.iloc[:,2].values
companies.drop(['Profit'],axis=1,inplace=True)
companie.drop(['Profit'],axis=1,inplace=True)
X=companies.iloc[:,:].values
X_T=companie.iloc[:,:].values

#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.006,random_state=0)
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X,y)
y_pred = model.predict(X)
########################## PCA #####################
c=0
for i in y_pred:
    c+=i
avg = c /len(y_pred)
c=0
for i in y_T:
    c+=i
avg1 = c /len(y_T)

print(avg)
#print(y_pred)
fig = plt.figure()
plt.plot(y,y_pred)
#print(model.coef_)# coefficient
#plt.plot(X,model.predict(X),color='k');plt.xlabel('x');plt.ylabel('y')
#print(model.intercept_)# intercepts
# calculating the  R squared error
#from sklearn.metrics import r2_score
#print(r2_score(y_test,y_pred))
print(X_T)
print(y_T)
print(avg/avg1)
################## Displays the window ################
while True:
    event, values = window.read()
    print(event, values)
    
    if event in  (None, 'Exit'):
        break
    
    if event == 'Profit preconizat':
        # Update the "output" text element
        # to be the value of average profit made 
        y_pred = model.predict(X_T)
        c=0
        for i in y_pred:
            c+=i
        avg = c /len(y_pred)
        window['-OUTPUT-'].update(avg)
    elif event == 'Rata profit':
        c=0
        for i in y_pred:
            c+=i
        avg = c /len(y_pred)
        c=0
        for i in y_T:
            c+=i
        avg1 = c /len(y_T)
        window['-OUTPUT-'].update(avg/avg1)
    elif event == 'Graph':
        
        draw_figure_w_toolbar(window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)
################## Displays the window ################

window.close()