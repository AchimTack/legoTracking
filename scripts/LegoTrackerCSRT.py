import cv2
import os
from datetime import datetime
import svgwrite

# Initialize the video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Variables to store the mask rectangle
drawing = False # true if mouse is pressed
ix, iy = -1, -1
mask_x, mask_y, mask_width, mask_height = -1, -1, -1, -1

# Mouse callback function to draw the mask
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, mask_x, mask_y, mask_width, mask_height

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_img = img.copy()
            cv2.rectangle(temp_img, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.putText(temp_img, 'Release to finalize rectangle', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Tracking', temp_img)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        mask_x, mask_y = min(ix, x), min(iy, y)
        mask_width, mask_height = abs(ix - x), abs(iy - y)
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)

# Read the first frame
success, img = cap.read()
cv2.putText(img, 'Draw a rectangle over the object, then press D', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.imshow('Tracking', img)
cv2.setMouseCallback('Tracking', draw_rectangle)

# Wait until the rectangle has been completed
while True:
    k = cv2.waitKey(1) & 0xFF
    if k == ord('d') and mask_width > 0 and mask_height > 0:
        break
    elif k == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

# Initialize the tracker with the first frame and the drawn mask
tracker = cv2.TrackerCSRT_create()
roi = img[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width]
bbox = cv2.selectROI('Tracking', roi, False)
tracker.init(roi, bbox)

path = []

def drawPath(img, path, color=(0, 255, 0), thickness=2):
    if len(path) > 1:
        for i in range(len(path) - 1):
            cv2.line(img, path[i], path[i + 1], color, thickness)

def drawBox(img, bbox, draw_rectangle=True):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    center = (int(x + w / 2) + mask_x, int(y + h / 2) + mask_y)
    path.append(center)
    if draw_rectangle:
        cv2.rectangle(img, (x + mask_x, y + mask_y), ((x + w + mask_x), (y + h + mask_y)), (255, 0, 255), 3, 1)
    drawPath(img, path)

while True:
    timer = cv2.getTickCount()
    success, img = cap.read()
    if not success:
        print("Error: Frame not read successfully.")
        break

    # Crop to the masked area
    mask = img[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width]
    success, bbox = tracker.update(mask)

    display_img = img.copy()
    if success:
        drawBox(display_img, bbox)
        cv2.putText(display_img, 'Tracking', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(display_img, 'Tracking lost', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(display_img, f'FPS: {int(fps)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow('Tracking', display_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        run_folder = f'runs/{timestamp}'
        if not os.path.exists(run_folder):
            os.makedirs(run_folder)
        desaturated_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        desaturated_img[..., 1] = desaturated_img[..., 1] * 0.2
        desaturated_img = cv2.cvtColor(desaturated_img, cv2.COLOR_HSV2BGR)
        drawPath(desaturated_img, path, color=(255, 0, 0), thickness=2)
        high_res_img = cv2.resize(desaturated_img, (desaturated_img.shape[1]*2, desaturated_img.shape[0]*2), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(f'{run_folder}/{timestamp}.jpg', high_res_img)
        # Save SVG
        dwg = svgwrite.Drawing(f'{run_folder}/{timestamp}.svg', size=(desaturated_img.shape[1], desaturated_img.shape[0]))
        if len(path) > 1:
            for i in range(len(path) - 1):
                dwg.add(dwg.line(start=path[i], end=path[i+1], stroke=svgwrite.rgb(0, 255, 0, '%')))
        dwg.save()
        break

cap.release()
cv2.destroyAllWindows()
