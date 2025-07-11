import tkinter

from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
import os
import sys

import mainNew

filename      = None
minuteDetail  = None
secondsSelect = None

def load () :
    global filename
    filename = filedialog.askopenfilename (
        title='Select File',
        initialdir = os.path.expanduser ('~')
    )
    if ((filename == None) or ((type (filename) is tuple)) or (filename == '')) :
        btnVideo['text'] = '...'
    else :
        btnVideo['text'] = os.path.basename (filename)

def execute () :
    global minuteDetail, secondsSelect
    minuteDetail  = entMinute.get ()
    secondsSelect = entSelArea.get ()

    if ((filename == None) or ((type(filename) is tuple)) or (filename == '')) :
        messagebox.showwarning ('Required Field', 'Select video file')
        btnVideo.focus ()
        return

    if (minuteDetail == '') :
        messagebox.showwarning ('Required Field', 'Fill "Analysis time step (in minute)" field')
        entMinute.focus ()
        return

    if (secondsSelect == '') :
        messagebox.showwarning ('Required Field', 'Fill "When to select area? (in seconds, separated by ,)" field')
        entSelArea.focus ()
        return


    window.withdraw ()
    mainNew.run ()


window = Tk ()
window.title('Heartbeat Analyzer')
window.geometry ('500x200')
window.resizable (0,0)

console = None

try:
    try:
        window.tk.call ('tk_getOpenFile', '-foobarbaz')
    except TclError:
        pass

    window.tk.call ('set', '::tk::dialog::file::showHiddenBtn', '1')
    window.tk.call ('set', '::tk::dialog::file::showHiddenVar', '0')
except:
    pass


lblVideo = Label (
    window,
    text = 'Select video file'
)


btnVideo = ttk.Button (
    window,
    text    = '...',
    command = load,
    style   = "File.TButton"
)

lblMinute = Label (
    window,
    text = 'Analysis time step (in minute)'
)

entMinute = Entry (window)

lblSelArea = Label (
    window,
    text = 'Fill "When to select area? (in seconds, separated by ,)'
)

entSelArea = Entry (window)

btnExecute = ttk.Button (
    window,
    text    = 'Execute',
    command = execute,
    style   = 'Execute.TButton'
)

lblVideo.grid   (row=0, column=0,padx=5,pady=10, sticky=W)
btnVideo.grid   (row=0, column=1, sticky=E+W)
lblMinute.grid  (row=1, column=0, padx=5,pady=5, columnspan=2, sticky=W)
entMinute.grid  (row=2, column=0, padx=5,pady=0, columnspan=2, sticky=W)
lblSelArea.grid (row=3, column=0, padx=5,pady=5, columnspan=2, sticky=W)
entSelArea.grid (row=4, column=0, padx=5,pady=0, columnspan=2, sticky=W)
btnExecute.place (relx=0.5, rely=0.85, anchor=CENTER)

style = ttk.Style()
style.theme_use ("alt")

style.map (
    "Execute.TButton",
    foreground = [
       ('!active','white'),
       ('pressed','#002e99'),
       ('active','#002e99'),
    ],
    background = [
       ('!active','#3689e6'),
       ('pressed','#64baff'),
       ('active','#64baff'),
    ]
)

style.map (
    "File.TButton",
    foreground = [
       ('!active','#fff'),
       ('pressed','#7a0000'),
       ('active','#7a0000'),
    ],
    background = [
       ('!active','#ed5353'),
       ('pressed','#ff8c82'),
       ('active','#ff8c82'),
    ]
)


# class PrintLogger():
#     def __init__(self, textbox):
#         self.textbox = textbox

#     def write(self, text):
#         self.textbox.insert(END, text)

#     def flush(self):
#         pass




