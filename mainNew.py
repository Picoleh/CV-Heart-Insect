import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from scipy.signal import find_peaks
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime


def getRoiArea(frame: np.array): # Get the coordinates of the ROI selected, returns None otherwise
    fig, ax = plt.subplots()
    ax.imshow(frame)
    coords = {}
    
    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata) # Click point
        x2, y2 = int(erelease.xdata), int(erelease.ydata)  # Release point

        # Guarantees that x1 < x2 and y1 < y2
        coords['x1'] = sorted([x1, x2])[0]
        coords['y1'] = sorted([y1, y2])[0]
        coords['x2'] = sorted([x1, x2])[1]
        coords['y2'] = sorted([y1, y2])[1]
    
    # Draws an rectangle on the selected area
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

def run_generator(video, fps: int, num_frames: int, coords: list,
                  showVideoMode: bool, callback=None): # Apply filters and crops to video/Send frames to Application show/Stores frames info
    
    mean_values = [] # Variable that stores the average brightness of the ROI
    frameCounter = int(video.get(cv2.CAP_PROP_POS_FRAMES))

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        if callback: # Calls the function to update the loading bar
            callback((frameCounter * 100.0) / num_frames)

        frameCounter += 1

        # Turns frame to grayscale and crops it
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_roi_gray = frame_gray[coords[1]:coords[3], coords[0]:coords[2]]

        #frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame_roi_gray = frame_gray[coords[1]:coords[3],coords[0]:coords[2]]

        # Expand regions to right and left of ROI (60) and aplies a binary threshold
        heart_roi_left = frame_gray[coords[1]:coords[3], coords[0] - 60: coords[0] + 60]
        _, heart_roi_left_thres = cv2.threshold(heart_roi_left, 127, 255, cv2.THRESH_BINARY)
        heart_roi_right = frame_gray[coords[1]:coords[3], coords[2] - 60: coords[2] + 60]
        _, heart_roi_right_thres = cv2.threshold(heart_roi_right, 127, 255, cv2.THRESH_BINARY)
        
        # Calculates the percentage of white pixels and the offset determined by it 
        bright_percen_left = np.count_nonzero(heart_roi_left_thres == 255) / heart_roi_left_thres.size
        offset_left = int(100 * (bright_percen_left - 0.5))
        bright_percen_right = np.count_nonzero(heart_roi_right_thres == 255) / heart_roi_right_thres.size
        offset_right = int(100 * (bright_percen_right - 0.5))
        
        # Turns frame back to BGR and draws the red bars
        frame_barra = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        frame_barra = cv2.rectangle(
            frame_barra,
            (coords[0] + offset_left, coords[1]),
            (coords[0] + offset_left + 5, coords[3]),
            (0,0,255), -1
        )
        
        frame_barra = cv2.rectangle(
            frame_barra,
            (coords[2] - offset_right,coords[1]),
            (coords[2] -offset_right - 5, coords[3]),
            (0,0,255), -1
        )

        # Stores the brightness of the ROI
        img_mean = np.mean(frame_roi_gray)
        mean_values.append(img_mean)

        yield frame_barra, frame_roi_gray, int(1000/fps) # Return the frame to be shown

    video.release()
    cv2.destroyAllWindows()

    if callback:
        callback(None, True)

    # Return the array with mean_values
    return np.array(mean_values)
    
def processar_resultados(mean_values: np.ndarray, timeMode: bool, num_frames: int, timeSelectArea: int, fps: int, step: int, fileName: str): # Calculates and shows information
    # Determines the graph title base on (timeMode = True [minutes], timeMode = False [seconds])
    title = "minutos" if timeMode else "segundos"
    
    # Uses a function that gathers peaks in a list of values
    #mean_values = np.array(mean_values)
    peaks, _ = find_peaks(mean_values, prominence=0.7)
    
    intensity_avg = np.mean(mean_values)
    greatest_intensity = np.max(mean_values)
    print(f"\nMedia intensidade ROI: {intensity_avg:.2f}")
    print(f"Maior intensidade ROI: {greatest_intensity:.2f}")
    
    duration_sec = int((num_frames - (timeSelectArea * fps)) / fps)
    bpm = (len(peaks) / duration_sec) * 60
    
    print(f"Segs: {duration_sec}, BPM: {bpm:.2f}")
    
    
    # Variables for step bpm analysis
    frames_por_step = step * fps
    bpms_steps = []
    step_times = []
    
    
    for i in range(0, num_frames, frames_por_step):
        #step_interval = mean_values[i : i + frames_por_step]
        
        peaks_local = peaks[(peaks >= i) & (peaks < i + frames_por_step)]
        
        bpm_local = (len(peaks_local) / step) * 60
        bpms_steps.append(bpm_local)
        
        step_time = i / fps
        timeOffset = timeSelectArea
        if timeMode:
            step_time /= 60
            timeOffset /= 60
        step_times.append(step_time + timeOffset)
    
    # Get frames to seconds array
    seconds = np.arange(num_frames) / fps
    peaks_seconds = (peaks / fps) + timeSelectArea
    
    # Graph sections
    fig1 = plt.figure()
    max_len = min(len(seconds[timeSelectArea * fps : ]), len(mean_values)) # Get the max len that seconds or mean_values can have (Fix missing frames)
    plt.plot(seconds[timeSelectArea * fps : (timeSelectArea * fps) + max_len], mean_values[: max_len])
    plt.plot(peaks_seconds, mean_values[peaks], 'ro', label="Picos")
    plt.xlabel("Segundos")
    plt.ylabel("Intensidade Média")
    plt.legend()
    
    if timeMode:
        step /= 60
    fig2 = plt.figure(figsize=(12,6))
    plt.bar(step_times, bpms_steps, width=step, align="edge", edgecolor="black")
    plt.xlabel('Tempo (início da janela)')
    plt.ylabel('BPM')
    plt.title(f'BPM por janela de {step} {title}')
    
    # Create PDF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"relatorio_{timestamp}.pdf"

    
    report_text = f"[Video Info]\nFile Name: {fileName} \nFPS: {fps} \nFrames: {num_frames} \nBPM: {bpm:.2f} \
    \nMedia de intensidade na ROI: {intensity_avg:.2f} \nMaior intensidade na ROI: {greatest_intensity:.2f} \n"
    with PdfPages(pdf_filename) as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig2)
        
        fig_text = plt.figure()
        plt.axis('off')
        plt.text(0.01, 0.95, report_text, ha='left', va='top', fontsize=14, wrap=True)
        pdf.savefig(fig_text)
        plt.close(fig_text)
        plt.close(fig1)
        plt.close(fig2)