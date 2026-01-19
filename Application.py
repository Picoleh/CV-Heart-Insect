import tkinter as tk

from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
import os
import cv2

import mainNew

# Vars init
filename      = None
minuteDetail  = None
startTime = None
endTime = None
console = None
timeState = False
showVideoState = True
videoOrientationState = True

root = Tk()
container = tk.Frame(root)
mainWindow = tk.Frame(container)

# Basic configs
root.title('Heartbeat Analyzer')
root.geometry ('480x270')
root.resizable(1,1)
container.pack(fill="both", expand=True)
loadingWindow = tk.Frame(container)
loadingWindow.place(relx=0, rely=0, relwidth=1.0, relheight=1.0)
mainWindow.place(relx=0, rely=0, relwidth=1.0, relheight=1.0)
mainWindow.tkraise()



# Commands
def load() : # Get file in os
    global filename
    filename = filedialog.askopenfilename(title='Select File', initialdir = os.path.expanduser('~'))
    if ((filename == None) or ((type(filename) is tuple)) or (filename == '')) :
        btnVideo['text'] = '...'
    else :
        btnVideo['text'] = os.path.basename(filename)

def execute() : # Handles the main execution of the program
    global minuteDetail, startTime, endTime
    minuteDetail  = entMinute.get()
    startTime = entStartTime.get()
    endTime = entEndTime.get()

    if ((filename == None) or ((type(filename) is tuple)) or (filename == '')) :
        messagebox.showwarning ('Required Field', 'Select video file')
        btnVideo.focus()
        return

    if (minuteDetail == '') :
        messagebox.showwarning ('Required Field', 'Fill "Analysis time step (in minute)" field')
        entMinute.focus()
        return

    if (startTime == '') :
        messagebox.showwarning ('Required Field', 'Fill "When to select area? (in seconds)" field')
        entStartTime.focus()
        return
    
    if(int(minuteDetail) <= 0):
        messagebox.showwarning ('Wrong Value', 'Inform a value > 0')
        entMinute.focus()
        return
    
    if(int(startTime) < 0):
        messagebox.showwarning ('Wrong Value', 'Inform a value >= 0')
        entStartTime.focus()
        return

    if(endTime !='' and int(endTime) < 0):
        messagebox.showwarning ('Wrong Value', 'Inform a value >= 0')
        entEndTime.focus()
        return

    # Get basic video info
    cap = cv2.VideoCapture(filename)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    startTimeFixed = 60 * int(startTime) if timeState else int(startTime)
    
    endTimeFixed = num_frames / fps

    if(endTime != ''):
        endTimeFixed = int(endTime)

    endTimeFixed = 60 * endTimeFixed if timeState else endTimeFixed
    endTimeFixed_frames = int(endTimeFixed * fps)
    startTimeFixed_frames = startTimeFixed * fps
    step = 60 * int(minuteDetail) if timeState else int(minuteDetail)
    frame_index = int(startTimeFixed * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Erro", "Não foi possível ler o frame para seleção de ROI.")
        return

    coords = mainNew.getRoiArea(frame)
    if not coords:
        messagebox.showwarning("Aviso", "Nenhuma ROI selecionada.")
        return

    # Shows loading window
    loadingWindow.tkraise()

    w = coords['x2'] - coords['x1']
    h = coords['y2'] - coords['y1']
    roi_tracker = (int(coords['x1']), int(coords['y1']), int(w), int(h))
    tracker = cv2.TrackerMIL_create()
    tracker.init(frame, roi_tracker)

    def callback(porcentagem=None, finished=False):
        if porcentagem is not None:
            atualizar_barra(porcentagem)
        if finished:
            mainWindow.tkraise()

    print(f"Start:{frame_index}({startTimeFixed})")
    print(f"End:{endTimeFixed_frames}({endTimeFixed})")

    gen = mainNew.run_generator(cap, fps, num_frames, coords, videoOrientationState, endTimeFixed_frames, tracker, callback)
    
    def show_next_frame():
        try:
            frame_barra, frame_roi, delay = next(gen)
            if showVideoState:
                cv2.imshow("Video", frame_barra)
                cv2.imshow("Regiao interesse", frame_roi)
                cv2.waitKey(delay)
            root.after(1, show_next_frame)
        except StopIteration as e:
            cv2.destroyAllWindows()
            mean_values = e.value  # Get appended values
            mainNew.processar_resultados(mean_values, timeState, startTimeFixed_frames, endTimeFixed_frames, fps, step, filename)

    show_next_frame()

def atualizar_barra(porcentagem):
    loadingBar['value'] = porcentagem
    lblPorcentagem['text'] = f"{porcentagem:.0f}%"
    root.update_idletasks()


def changeTimeState():
    global timeState
    timeState = not timeState
    if timeState:
        btnTime.config(text="Minutes")
    else:
        btnTime.config(text="Seconds")
    
def changeShowVideoState():
    global showVideoState
    showVideoState = not showVideoState
    if showVideoState:
        btnShowVideo.config(text="Show Video")
    else:
        btnShowVideo.config(text="Hide Video")

def changeVideoOrientationState():
    global videoOrientationState
    videoOrientationState = not videoOrientationState
    if videoOrientationState:
        btnVideoOrientation['text'] = "Horizontal"
    else:
        btnVideoOrientation['text'] = "Vertical"

def on_close():
    print("Encerrando...")
    root.destroy()
    root.quit()

try:
    try:
        mainWindow.tk.call('tk_getOpenFile', '-foobarbaz')
    except TclError:
        pass

    mainWindow.tk.call('set', '::tk::dialog::file::showHiddenBtn', '1')
    mainWindow.tk.call('set', '::tk::dialog::file::showHiddenVar', '0')
except:
    pass
# --------
# Widgets
lblVideo = Label(
    mainWindow,
    text = 'Select video file'
)

btnVideo = ttk.Button(
    mainWindow,
    text    = '...',
    command = load,
    style   = "File.TButton"
)

lblTime = Label(
    mainWindow,
    text = 'Time dimension'
)

lblShowVideo = Label(
    mainWindow,
    text = 'Video display'
)

btnTime = ttk.Button(
    mainWindow,
    text="Seconds",
    command=changeTimeState
    )

btnShowVideo = ttk.Button(
    mainWindow,
    text="Show Video",
    command=changeShowVideoState,
    )

lblVideoOrientation = Label(mainWindow, text="Video Orientation")

btnVideoOrientation = ttk.Button(
    mainWindow,
    text= "Horizontal",
    command=changeVideoOrientationState
)

lblTimeStep = Label(mainWindow, text='Analysis time step')

entMinute = Entry(mainWindow)

lblStartTime = Label(mainWindow, text='Start Analysis')

entStartTime = Entry(mainWindow)

lblEndTime = Label(mainWindow, text='End Analysis')

entEndTime = Entry(mainWindow)

btnExecute = ttk.Button(
    mainWindow,
    text    = 'Execute',
    command = execute,
    style   = 'Execute.TButton'
)

loadingBar = ttk.Progressbar(loadingWindow, length=300, mode='determinate', maximum=100)
lblPorcentagem = tk.Label(loadingWindow, text="0%")

# --------
# Layout 
lblVideo.place(relx=0.05, rely=0.05, relwidth=0.30, relheight=0.1)
btnVideo.place(relx=0.35, rely=0.05, relwidth=0.6, relheight=0.1)

lblTime.place(relx=0.05, rely=0.2, relwidth=0.28, relheight=0.1)
lblShowVideo.place(relx=0.36, rely=0.2, relwidth=0.28, relheight=0.1)
lblVideoOrientation.place(relx=0.67, rely=0.2, relwidth=0.28, relheight=0.1)

btnTime.place(relx=0.05, rely=0.35, relwidth=0.28, relheight=0.1)
btnShowVideo.place(relx=0.36, rely=0.35, relwidth=0.28, relheight=0.1)
btnVideoOrientation.place(relx=0.67, rely=0.35, relwidth=0.28, relheight=0.1)

lblTimeStep.place(relx=0.05, rely=0.5, relwidth=0.28, relheight=0.1)
lblStartTime.place(relx=0.36, rely=0.5, relwidth=0.28, relheight=0.1)
lblEndTime.place(relx=0.67, rely=0.5, relwidth=0.28, relheight=0.1)

entMinute.place(relx=0.05, rely=0.65, relwidth=0.28, relheight=0.07)
entStartTime.place(relx=0.36, rely=0.65, relwidth=0.28, relheight=0.07)
entEndTime.place(relx=0.67, rely=0.65, relwidth=0.28, relheight=0.07)

btnExecute.place(relx=0.7, rely=0.85, relwidth=0.25, relheight=0.1)
# |||||||||
loadingBar.place(relx=0.1, rely=0.45, relwidth=0.8, relheight=0.1)
lblPorcentagem.place(relx=0.4, rely=0.6, relwidth=0.2, relheight=0.07)
# --------

# Styles
style = ttk.Style()
style.theme_use ("alt")

style.map(
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

style.map(
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
# --------

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()