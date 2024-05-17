import os
import datetime
from tkinter import *
from PIL import ImageTk, Image
from tkinter import font, filedialog
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

CONFIDENCE_THRESHOLD = 0.8
Blue = (0, 0, 255)

cap = cv2.VideoCapture(0)

root = Tk()
root.geometry('1000x700')
root.resizable(0, 0)
root.title('DroneVision')
root.iconbitmap('New folder/drone1.ico')
root.configure(background='gray')

# GUI setup
img = Image.open('New folder/drone1.png')
resized_img = img.resize((195, 195))
img = ImageTk.PhotoImage(resized_img)
img_label = Label(root, image=img, bg='gray')
img_label.pack()

img1 = Image.open('New folder/drone1.png')
resized_img1 = img1.resize((175, 175))
img1 = ImageTk.PhotoImage(resized_img1)

custom_font1_path = 'C:/Users/mukul/AppData/Local/Microsoft/Windows/Fonts/Komika Axis.ttf'
custom_font1 = font.Font(family='Komika Axis', size=15)

custom_font2_path = 'C:/Users/mukul/AppData/Local/Microsoft/Windows/Fonts/Exo 2 Black.ttf'
custom_font2 = font.Font(family='Exo 2 Black', size=27)

custom_font3_path = 'C:/Users/mukul/AppData/Local/Microsoft/Windows/Fonts/Bahnschrift.ttf'
custom_font3 = font.Font(family='Bahnschrift SemiLight Condensed', size=23)

custom_font4_path = 'C:/Users/mukul/AppData/Local/Microsoft/Windows/Fonts/Komika Axis.ttf'
custom_font4 = font.Font(family='Komika Axis', size=8)

text_label = Label(root, text='DroneVision', fg='black', bg='gray')
text_label.pack()
text_label.config(font=custom_font2)

text_label2 = Label(root, text='Choose from the below options', fg='black', bg='gray')
text_label2.pack(pady=30)
text_label2.config(font=custom_font3)

frame = Frame(root)
frame.pack(pady=87)
frame.config(bg='gray')

button1 = Button(frame, text='Live Preview', bg='#67d796', fg='black', width=20, height=2, command=lambda: open_top())
button1.pack(side=LEFT, padx=60, pady=(10, 5))
button1.config(font=custom_font1)

button2 = Button(frame, text='Recordings', bg='#67d796', fg='black', width=20, height=2)
button2.pack(side=LEFT, padx=60, pady=(10, 5))
button2.config(font=custom_font1)

def open_top():
    global cap
    top_window = Toplevel(root)
    top_window.geometry('640x770')
    top_window.configure(background='gray')
    top_window.title("DroneVision")
    top_window.resizable(0, 0)
    top_window.iconbitmap('New folder/drone1.ico')

    img_label1 = Label(top_window, image=img1, bg='gray')
    img_label1.pack()

    text_label0 = Label(top_window, text='DroneVision', fg='black', bg='gray')
    text_label0.pack(pady=0)
    text_label0.config(font=custom_font2)

    video_label = Label(top_window)
    video_label.pack()

    button_frame = Frame(top_window)
    button_frame.pack()

    capture_button = Button(button_frame, text="Capture", bg='#67d796', fg='black', width=20, height=3, command=capture)
    capture_button.pack()
    capture_button.config(font=custom_font4)

    # global record_button
    # record_button = Button(button_frame, text="Record", bg='#67d796', fg='black', width=20, height=3, command=toggle_record)
    # record_button.pack(side=LEFT)
    # record_button.config(font=custom_font4)

    cap = cv2.VideoCapture(1)
    motion_detected = False
    frame1 = None

    def update_video_label():
        
        nonlocal motion_detected, frame1
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if frame1 is None:
                frame1 = gray
            else:
                frame2 = gray

                diff = cv2.absdiff(frame1, frame2)
                thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)

                contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                   if cv2.contourArea(contour) < 1000:
                    continue
                cv2.putText(frame, "Motion Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                

                start = datetime.datetime.now()
                detections = model(frame)[0]
                for data in detections.boxes.data.tolist():
                    confidence = data[4]
                    if float(confidence) < CONFIDENCE_THRESHOLD:
                        continue
                    xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                    class_label = model.names[int(data[5])]
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), Blue, 2)
                    cv2.putText(frame, class_label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, Blue, 2)
                end = datetime.datetime.now()
                total = (end - start).total_seconds()
                print(f"Time to process 1 frame: {total * 1000:.0f} milliseconds")

                frame1 = frame2

            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            video_label.configure(image=photo)
            video_label.image = photo
            video_label.after(10, update_video_label)

    update_video_label()

def capture():
    global cap
    ret, frame = cap.read()
    if ret:
        filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if filename:
            cv2.imwrite(filename, frame)

# def toggle_record():
#     global is_recording
#     if not is_recording:
#         start_recording()
#     else:
#         stop_recording()

# def start_recording():
#     global is_recording, out, record_button
#     is_recording = True
#     record_button.config(text="Stop Recording", bg='red', fg='white')
#     filename = f"output.mp4"

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')

#     out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

# def stop_recording():
#     global is_recording, out, record_button
#     is_recording = False
#     record_button.config(text="Record", bg='#67d796', fg='black')

#     out.release()

# def save_video():
#     global out
#     filename = filedialog.asksaveasfilename(defaultextension="*.mp4", filetypes=[("MP4 files", "*.mp4")])
#     if filename:
#         print("Saving video to:", filename)

#         if os.path.exists('output.mp4'):
#             import shutil
#             shutil.move('output.mp4', filename)
#             print("Video saved successfully and moved to:", filename)
#         else:
#             print("Error: No video file found to save.")

def capture():
    global cap
    ret, frame = cap.read()
    if ret:
        filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if filename:
            cv2.imwrite(filename, frame)
root.mainloop()