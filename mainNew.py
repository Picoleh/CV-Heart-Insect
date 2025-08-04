# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 19:08:23 2025

@author: LucasTrevizanPícoli
"""
import Application
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
import matplotlib.image as mpimg
from scipy.signal import find_peaks

import Application

def readFileInfo(path : str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
    print("[ Video info ]")
    print("Name: {}".format(path))
    print("FPS: {}".format(fps))
    print("Frames: {}".format(num_frames))
    seg = num_frames/fps
    minu = int(seg/60)
    seg = int(seg % 60)
    print("Duração: {}m{}s".format(minu, seg))
    return cap, fps, num_frames, int(Application.secondsSelect)

def getRoiArea(frame: np.array):    
    fig, ax = plt.subplots()
    ax.imshow(frame)
    coords = {}
    
    
    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata) # ponto onde clicou
        x2, y2 = int(erelease.xdata), int(erelease.ydata)  # ponto onde soltou
        coords['x1'] = sorted([x1, x2])[0]
        coords['y1'] = sorted([y1, y2])[0]
        coords['x2'] = sorted([x1, x2])[1]
        coords['y2'] = sorted([y1, y2])[1]
        # print(f"Retângulo: ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f})")
    
    toggle_selector = RectangleSelector(ax, onselect,
                                    useblit=True,
                                    button=[1],  # botão esquerdo do mouse
                                    minspanx=5, minspany=5,
                                    spancoords='pixels',
                                    interactive=True)
    plt.show(block=True)
    
    if coords:
        return coords['x1'], coords['y1'], coords['x2'], coords['y2']
    else:
        return None
    
def run():
    
    # Criacao variaveis
    name = Application.filename
    video, fps, n_frames, timeSelectArea = readFileInfo(name)
    
    coords = []
    mean_values = []
    frame_log = []
    frameCounter = 0
    # Processamento
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        # Porcentagem
        print(f"\rProcessando: {(frameCounter * 100.0) / n_frames:.2f}%", end="")
        frameCounter += 1
        
        if(frameCounter < (timeSelectArea * fps) + 1):
            continue
        if(frameCounter == (timeSelectArea * fps) + 1):
            coords = getRoiArea(frame)
        
        
        # Filtros e cortes
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_roi_gray = frame_gray[coords[1]:coords[3],coords[0]:coords[2]]
        
        
        # Barras coração
        frame_barra = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        frame_barra = cv2.rectangle(frame_barra, (coords[0], coords[1]), (coords[0] + 5, coords[3]), (0,0,255), -1)
        frame_barra = cv2.rectangle(frame_barra, (coords[2], coords[1]), (coords[2] - 5, coords[3]), (0,0,255), -1)
        
        # Exibicao
        cv2.imshow("Video", frame_barra)
        cv2.imshow("Regiao interesse", frame_roi_gray)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        
        # Contagem
        img_mean = np.mean(frame_roi_gray)
        mean_values.append(img_mean)
        
        # Historico
        frame_log.append(frame_roi_gray)
        
    video.release()
    cv2.destroyAllWindows()
    
    frame_log = np.array(frame_log)
    mean_values = np.array(mean_values)
    peaks, _ = find_peaks(mean_values, prominence=0.7)
    
    
    # Calculo BPM
    print(f"\nMedia intensidade ROI: {np.mean(mean_values)}")
    print(f"Maior intensidade ROI: {np.max(mean_values)}")
    
    duration_sec = (n_frames - (timeSelectArea * fps)) / fps
    bpm = (len(peaks) / duration_sec) * 60
    
    print(f"Segs: {duration_sec}, BPM: {bpm}")
    
    # Transformar de frames para segundo
    seconds = np.arange(n_frames) / fps
    peaks_seconds = (peaks / fps) + timeSelectArea
    
    # Exibir gráfico
    plt.plot(seconds[timeSelectArea * fps:], mean_values)
    plt.plot(peaks_seconds, mean_values[peaks], 'ro', label="Picos")
    plt.xlabel("Segundos")
    plt.ylabel("Intensidade Média")
    plt.legend()
    plt.show()
    
    Application.window.quit()
    Application.window.destroy()

def main():
    Application.window.mainloop()

if __name__ == "__main__":
    main()