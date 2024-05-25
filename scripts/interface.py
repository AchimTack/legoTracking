from functions import undistort_and_track
from tkinter import *

root = Tk()

# VARIABLES


# Set mat dimensions in cm
matLength = 236
matWidth = 114

# Set Webcam Parameters
cam_id = 1
frame_width = 1920
frame_height = 1080

# Define which marker range is to be tracked (minimizes false-positives)
marker_ids_to_track = set(range(1, 10))
edge_marker_ids = [91, 92, 93, 94]

# Define export parameters (1=true, 0=false)
export_jpg = IntVar()
export_svg = IntVar()
export_csv = IntVar()
export_mp4 = IntVar()
img_output_width = 1000




root.title('LegoTracker Arucode')
root.configure(background='white')

title = Label(root, text='Lego Arucode Tracker')
btn_export_jpg = Checkbutton(root, text='Export JPG', variable=export_jpg, onvalue=1, offvalue=0)
btn_export_svg = Checkbutton(root, text='Export SVG', variable=export_svg, onvalue=1, offvalue=0)
btn_export_csv = Checkbutton(root, text='Export CSV', variable=export_csv, onvalue=1, offvalue=0)
btn_export_mp4 = Checkbutton(root, text='Export MP4', variable=export_mp4, onvalue=1, offvalue=0)

title.pack()
btn_export_jpg.pack()
btn_export_svg.pack()
btn_export_csv.pack()
btn_export_mp4.pack()
root.mainloop()

print(export_svg.get(), export_jpg.get(), export_mp4.get(), export_csv.get())
export_svg, export_jpg, export_mp4, export_csv = export_svg.get(), export_jpg.get(), export_mp4.get(), export_csv.get()
print(export_svg, export_jpg, export_mp4, export_csv)


undistort_and_track(matLength, matWidth,
                        marker_ids_to_track, edge_marker_ids,
                        cam_id, frame_width, frame_height, img_output_width,
                        export_jpg, export_svg, export_csv, export_mp4)
