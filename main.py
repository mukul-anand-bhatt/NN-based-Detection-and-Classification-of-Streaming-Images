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


# GUI setup

root = Tk()
root.geometry('1000x700')
root.resizable(0, 0)
root.title('DroneVision')
root.iconbitmap('NN_Based_Detection_and_Classification_of_Streaming_Images/Logo/Logo-ICO.ico') # Change the file location
root.configure(background='gray')

img = Image.open('NN_Based_Detection_and_Classification_of_Streaming_Images/Logo/Logo-PNG.png') # Change the file location
resized_img = img.resize((195, 195))
img = ImageTk.PhotoImage(resized_img)
img_label = Label(root, image=img, bg='gray')
img_label.pack()

img1 = Image.open('NN_Based_Detection_and_Classification_of_Streaming_Images/Logo/Logo-PNG.png') # Change the file location
resized_img1 = img1.resize((175, 175))
img1 = ImageTk.PhotoImage(resized_img1)

custom_font1_path = 'C:/Users/mukul/AppData/Local/Microsoft/Windows/Fonts/Komika Axis.ttf' # Change the file location
custom_font1 = font.Font(family='Komika Axis', size=15)

custom_font2_path = 'C:/Users/mukul/AppData/Local/Microsoft/Windows/Fonts/Exo 2 Black.ttf' # Change the file location
custom_font2 = font.Font(family='Exo 2 Black', size=27)

custom_font3_path = 'C:/Users/mukul/AppData/Local/Microsoft/Windows/Fonts/Bahnschrift.ttf' # Change the file location
custom_font3 = font.Font(family='Bahnschrift SemiLight Condensed', size=23)

custom_font4_path = 'C:/Users/mukul/AppData/Local/Microsoft/Windows/Fonts/Komika Axis.ttf' # Change the file location
custom_font4 = font.Font(family='Komika Axis', size=8)

text_label = Label(root, text='DroneVision', fg='black', bg='gray')
text_label.pack()
text_label.config(font=custom_font2)

text_label2 = Label(root, text='Click below to watch live preview', fg='black', bg='gray')
text_label2.pack(pady=30)
text_label2.config(font=custom_font3)

frame = Frame(root)
frame.pack(pady=87)
frame.config(bg='gray')

button1 = Button(frame, text='Live Preview', bg='#67d796', fg='black', width=20, height=2, command=lambda: open_top())
button1.pack(side=LEFT, padx=60, pady=(10, 5))
button1.config(font=custom_font1)

def open_top():
    global cap
    top_window = Toplevel(root)
    top_window.geometry('640x770')
    top_window.configure(background='gray')
    top_window.title("DroneVision")
    top_window.resizable(0, 0)
    top_window.iconbitmap('NN_Based_Detection_and_Classification_of_Streaming_Images/Logo/Logo-ICO.ico') # Change the file location

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

    cap = cv2.VideoCapture(0)  # Change the camera index if necessary

    def update_video_label():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)


            # Motion Detection

            if not hasattr(update_video_label, "frame1"):
                update_video_label.frame1 = gray
            else:
                frame2 = gray
                diff = cv2.absdiff(update_video_label.frame1, frame2)
                thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) < 1000:
                        continue
                    cv2.putText(frame, "Motion Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                update_video_label.frame1 = frame2


            # Object Detection

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

            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            video_label.configure(image=photo)
            video_label.image = photo

        video_label.after(10, update_video_label)

    update_video_label()


# Save image

def capture():
    global cap
    ret, frame = cap.read()
    if ret:
        filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if filename:
            cv2.imwrite(filename, frame)

root.mainloop()