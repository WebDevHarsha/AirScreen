import cv2
import mediapipe as mp
import numpy as np
import os
import math
import json
import time
from collections import deque
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS, 5)
width = 1280
height = 720
cap.set(3, width)
cap.set(4, height)

imgCanvas = np.zeros((height, width, 3), np.uint8)

# Store drawing strokes for shape analysis
current_stroke = []
all_strokes = []
undo_stack = []  # NEW: For undo functionality
stroke_timeout = 30  # frames without drawing before considering stroke complete

# NEW: Additional colors and features
colors = [
    (0, 0, 255),    # Red
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 0),      # Black (eraser)
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 128),  # Purple
]

# NEW: Gesture detection variables
gesture_start_time = 0
gesture_threshold = 1.0  # seconds to hold gesture
show_gesture_feedback = False
gesture_feedback_text = ""

folderPath = 'Header'
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

header = overlayList[0]
drawColor = (0, 0, 255)
thickness = 20 
tipIds = [4, 8, 12, 16, 20] 
xp, yp = [0, 0] 
frame_count = 0
last_draw_frame = 0
color_index = 0  # NEW: Track current color

# NEW: Statistics tracking
stats = {
    'total_strokes': 0,
    'lines': 0,
    'circles': 0,
    'rectangles': 0,
    'freeform': 0,
    'session_start': time.time()
}

def detect_line(points, tolerance=0.95):
    """Detect if points form a straight line using linear regression"""
    if len(points) < 10:
        return False, None, None
    
    points = np.array(points)
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    
    # Calculate R-squared score
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    if r2 > tolerance:
        return True, points[0], points[-1]
    return False, None, None

def detect_circle(points, tolerance=0.8):
    """Detect if points form a circle"""
    if len(points) < 20:
        return False, None, None
    
    points = np.array(points)
    
    # Check if the stroke forms a closed loop
    start_point = points[0]
    end_point = points[-1]
    distance_to_close = np.sqrt((start_point[0] - end_point[0])**2 + (start_point[1] - end_point[1])**2)
    
    # Must be reasonably closed
    avg_distance = np.mean([np.sqrt((points[i][0] - points[i+1][0])**2 + (points[i][1] - points[i+1][1])**2) 
                           for i in range(len(points)-1)])
    
    if distance_to_close > avg_distance * 5:
        return False, None, None
    
    # Fit circle using least squares
    try:
        # Convert to numpy array
        x = points[:, 0]
        y = points[:, 1]
        
        # Circle fitting using algebraic method
        A = np.c_[x, y, np.ones(len(x))]
        b = x**2 + y**2
        c = np.linalg.lstsq(A, b, rcond=None)[0]
        
        center_x = c[0] / 2
        center_y = c[1] / 2
        radius = np.sqrt(c[2] + center_x**2 + center_y**2)
        
        # Check how well points fit the circle
        distances = [abs(np.sqrt((p[0] - center_x)**2 + (p[1] - center_y)**2) - radius) for p in points]
        avg_error = np.mean(distances)
        
        # If error is reasonable, it's a circle
        if avg_error < radius * (1 - tolerance):
            return True, (int(center_x), int(center_y)), int(radius)
    except:
        pass
    
    return False, None, None

def detect_rectangle(points, tolerance=0.9):
    """Detect if points form a rectangle"""
    if len(points) < 20:
        return False, None
    
    points = np.array(points)
    
    # Check if stroke is closed
    start_point = points[0]
    end_point = points[-1]
    distance_to_close = np.sqrt((start_point[0] - end_point[0])**2 + (start_point[1] - end_point[1])**2)
    
    avg_distance = np.mean([np.sqrt((points[i][0] - points[i+1][0])**2 + (points[i][1] - points[i+1][1])**2) 
                           for i in range(len(points)-1)])
    
    if distance_to_close > avg_distance * 5:
        return False, None
    
    # Find corner points using angle changes
    angles = []
    for i in range(1, len(points)-1):
        v1 = points[i] - points[i-1]
        v2 = points[i+1] - points[i]
        angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8), -1.0, 1.0))
        angles.append(abs(angle))
    
    # Find points with significant angle changes (corners)
    corner_threshold = np.pi / 3  # 60 degrees
    corners = []
    for i, angle in enumerate(angles):
        if angle > corner_threshold:
            corners.append(points[i+1])
    
    if len(corners) == 4:
        # Sort corners to form rectangle
        corners = np.array(corners)
        # Simple rectangle approximation
        min_x, min_y = np.min(corners, axis=0)
        max_x, max_y = np.max(corners, axis=0)
        rect_corners = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
        return True, rect_corners
    
    return False, None

def smooth_stroke(points, smoothing_factor=0.3):
    """Apply smoothing to a stroke"""
    if len(points) < 3:
        return points
    
    smoothed = [points[0]]
    for i in range(1, len(points)-1):
        # Weighted average with neighbors
        prev_point = np.array(points[i-1])
        curr_point = np.array(points[i])
        next_point = np.array(points[i+1])
        
        smoothed_point = (1-smoothing_factor) * curr_point + smoothing_factor * (prev_point + next_point) / 2
        smoothed.append(smoothed_point.astype(int))
    
    smoothed.append(points[-1])
    return smoothed

def process_completed_stroke(stroke, canvas, color, thick):
    """Process a completed stroke and apply corrections"""
    if len(stroke) < 5:
        return 'freeform'
    
    # Try to detect shapes in order of specificity
    is_line, start_pt, end_pt = detect_line(stroke)
    is_circle, center, radius = detect_circle(stroke)
    is_rect, corners = detect_rectangle(stroke)
    
    if is_line and start_pt is not None and end_pt is not None:
        # Draw straight line
        cv2.line(canvas, tuple(start_pt), tuple(end_pt), color, thick)
        print("Corrected line detected!")
        return 'line'
        
    elif is_circle and center is not None and radius is not None:
        # Draw perfect circle
        cv2.circle(canvas, center, radius, color, thick)
        print("Corrected circle detected!")
        return 'circle'
        
    elif is_rect and corners is not None:
        # Draw rectangle
        pts = np.array(corners, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], True, color, thick)
        print("Corrected rectangle detected!")
        return 'rectangle'
        
    else:
        # Apply smoothing to freeform drawing
        smoothed_stroke = smooth_stroke(stroke)
        for i in range(len(smoothed_stroke) - 1):
            cv2.line(canvas, tuple(smoothed_stroke[i]), tuple(smoothed_stroke[i + 1]), color, thick)
        return 'freeform'

# NEW: Save and load functions
def save_drawing(strokes, filename="drawing.json"):
    """Save the current drawing to a file"""
    try:
        data = {
            'strokes': strokes,
            'stats': stats,
            'timestamp': time.time()
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Drawing saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving drawing: {e}")
        return False

def load_drawing(filename="drawing.json"):
    """Load a drawing from a file"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data.get('strokes', [])
    except Exception as e:
        print(f"Error loading drawing: {e}")
        return []

# NEW: Undo functionality
def undo_last_stroke():
    """Remove the last drawn stroke"""
    global all_strokes, undo_stack, imgCanvas
    if all_strokes:
        # Move last stroke to undo stack
        undo_stack.append(all_strokes.pop())
        
        # Redraw canvas without the last stroke
        imgCanvas = np.zeros((height, width, 3), np.uint8)
        for stroke_data in all_strokes:
            process_completed_stroke(stroke_data['points'], imgCanvas, 
                                   stroke_data['color'], stroke_data['thickness'])
        print("Stroke undone!")

# NEW: Redo functionality
def redo_last_stroke():
    """Restore the last undone stroke"""
    global all_strokes, undo_stack, imgCanvas
    if undo_stack:
        # Move stroke back from undo stack
        stroke_data = undo_stack.pop()
        all_strokes.append(stroke_data)
        
        # Redraw the stroke
        process_completed_stroke(stroke_data['points'], imgCanvas, 
                               stroke_data['color'], stroke_data['thickness'])
        print("Stroke redone!")

# NEW: Display statistics and controls
def draw_ui_elements(image):
    """Draw UI elements like statistics and controls"""
    global stats, show_gesture_feedback, gesture_feedback_text
    
    # Display current color indicator
    cv2.circle(image, (50, 150), 20, drawColor, -1)
    cv2.circle(image, (50, 150), 20, (255, 255, 255), 2)
    
    # Display thickness indicator
    cv2.circle(image, (50, 200), thickness, drawColor, 2)
    
    # Display statistics
    session_time = int(time.time() - stats['session_start'])
    stats_text = [
        f"Session: {session_time//60}m {session_time%60}s",
        f"Total: {stats['total_strokes']}",
        f"Lines: {stats['lines']}",
        f"Circles: {stats['circles']}",
        f"Rectangles: {stats['rectangles']}",
        f"Freeform: {stats['freeform']}"
    ]
    
    for i, text in enumerate(stats_text):
        cv2.putText(image, text, (width - 200, 150 + i * 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display gesture feedback
    if show_gesture_feedback:
        cv2.putText(image, gesture_feedback_text, (width//2 - 100, height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Display controls
    controls_text = [
        "Controls:",
        "Peace Sign: Select/Navigate",
        "Index Finger: Draw",
        "Index + Pinky: Erase",
        "Fist: Clear All",
        "Thumbs Up (hold): Undo",
        "OK Sign (hold): Redo",
        "Three Fingers (hold): Save"
    ]
    
    for i, text in enumerate(controls_text):
        color = (0, 255, 255) if i == 0 else (255, 255, 255)
        cv2.putText(image, text, (10, height - 200 + i * 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

# NEW: Gesture detection for special commands
def detect_special_gestures(fingers, current_time):
    """Detect special gestures for commands"""
    global gesture_start_time, show_gesture_feedback, gesture_feedback_text
    
    # Thumbs up gesture for undo
    if fingers == [1, 0, 0, 0, 0]:
        if gesture_start_time == 0:
            gesture_start_time = current_time
            show_gesture_feedback = True
            gesture_feedback_text = "Hold for Undo..."
        elif current_time - gesture_start_time > gesture_threshold:
            undo_last_stroke()
            gesture_start_time = 0
            show_gesture_feedback = False
            return True
    
    # OK sign (thumb + index) for redo
    elif fingers == [1, 1, 0, 0, 0]:
        if gesture_start_time == 0:
            gesture_start_time = current_time
            show_gesture_feedback = True
            gesture_feedback_text = "Hold for Redo..."
        elif current_time - gesture_start_time > gesture_threshold:
            redo_last_stroke()
            gesture_start_time = 0
            show_gesture_feedback = False
            return True
    
    # Three fingers for save
    elif fingers == [0, 1, 1, 1, 0]:
        if gesture_start_time == 0:
            gesture_start_time = current_time
            show_gesture_feedback = True
            gesture_feedback_text = "Hold to Save..."
        elif current_time - gesture_start_time > gesture_threshold:
            save_drawing(all_strokes, f"drawing_{int(time.time())}.json")
            gesture_start_time = 0
            show_gesture_feedback = False
            return True
    
    else:
        # Reset gesture timing if not holding a gesture
        if gesture_start_time != 0:
            gesture_start_time = 0
            show_gesture_feedback = False
    
    return False

with mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        frame_count += 1
        current_time = time.time()
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        
        image.flags.writeable = False
        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        is_drawing = False
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                points = []
                for lm in hand_landmarks.landmark:
                    points.append([int(lm.x * width), int(lm.y * height)])

                if len(points) != 0:
                    x1, y1 = points[8]  
                    x2, y2 = points[12] 
                    x3, y3 = points[4]  
                    x4, y4 = points[20] 

                    fingers = []
                    
                    if points[tipIds[0]][0] < points[tipIds[0] - 1][0]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                    for id in range(1, 5):
                        if points[tipIds[id]][1] < points[tipIds[id] - 2][1]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                    # NEW: Check for special gestures first
                    if detect_special_gestures(fingers, current_time):
                        continue

                    # Selection mode
                    nonSel = [0, 3, 4] 
                    if (fingers[1] and fingers[2]) and all(fingers[i] == 0 for i in nonSel):
                        xp, yp = [x1, y1]

                        if(y1 < 125):
                            # NEW: Extended color selection
                            color_sections = [
                                (170, 295, 0),   # Red
                                (295, 420, 1),   # Blue  
                                (420, 545, 2),   # Green
                                (545, 670, 3),   # Black
                                (670, 795, 4),   # Cyan
                                (795, 920, 5),   # Magenta
                                (920, 1045, 6),  # Yellow
                                (1045, 1170, 7), # Purple
                            ]
                            
                            for start_x, end_x, color_idx in color_sections:
                                if start_x < x1 < end_x and color_idx < len(colors):
                                    header = overlayList[min(color_idx, len(overlayList)-1)]
                                    drawColor = colors[color_idx]
                                    color_index = color_idx
                                    break

                        cv2.rectangle(image, (x1-10, y1-15), (x2+10, y2+23), drawColor, cv2.FILLED)

                    # Eraser mode
                    nonStand = [0, 2, 3] 
                    if (fingers[1] and fingers[4]) and all(fingers[i] == 0 for i in nonStand):
                        cv2.line(image, (xp, yp), (x4, y4), drawColor, 5) 
                        xp, yp = [x1, y1]

                    # Drawing mode
                    nonDraw = [0, 2, 3, 4]
                    if fingers[1] and all(fingers[i] == 0 for i in nonDraw):
                        is_drawing = True
                        last_draw_frame = frame_count
                        
                        cv2.circle(image, (x1, y1), int(thickness/2), drawColor, cv2.FILLED) 
                        if xp==0 and yp==0:
                            xp, yp = [x1, y1]
                        
                        # Add point to current stroke
                        current_stroke.append([x1, y1])
                        
                        # Draw on canvas (temporary, will be replaced by corrected version)
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
                        
                        xp, yp = [x1, y1]

                    # Clear canvas
                    if all(fingers[i] == 0 for i in range(0, 5)):
                        imgCanvas = np.zeros((height, width, 3), np.uint8)
                        current_stroke = []
                        all_strokes = []
                        undo_stack = []  # NEW: Clear undo stack
                        # NEW: Reset stats
                        stats.update({
                            'total_strokes': 0, 'lines': 0, 'circles': 0, 
                            'rectangles': 0, 'freeform': 0
                        })
                        xp, yp = [x1, y1]

                    # Thickness adjustment
                    selecting = [1, 1, 0, 0, 0] 
                    setting = [1, 1, 0, 0, 1]   
                    if all(fingers[i] == j for i, j in zip(range(0, 5), selecting)) or all(fingers[i] == j for i, j in zip(range(0, 5), setting)):

                        r = int(math.sqrt((x1-x3)**2 + (y1-y3)**2)/3)
                        x0, y0 = [(x1+x3)/2, (y1+y3)/2]
                        v1, v2 = [x1 - x3, y1 - y3]
                        v1, v2 = [-v2, v1]

                        mod_v = math.sqrt(v1**2 + v2**2)
                        if mod_v > 0:  # NEW: Avoid division by zero
                            v1, v2 = [v1/mod_v, v2/mod_v]
                        
                        c = 3 + r
                        x0, y0 = [int(x0 - v1*c), int(y0 - v2*c)]
                        cv2.circle(image, (x0, y0), int(r/2), drawColor, -1)

                        if fingers[4]:                        
                            thickness = max(5, min(50, r))  # NEW: Limit thickness range
                            cv2.putText(image, 'Check', (x4-25, y4-8), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0,0,0), 1)

                        xp, yp = [x1, y1]

        # Check if stroke is complete (no drawing for several frames)
        if not is_drawing and current_stroke and frame_count - last_draw_frame > stroke_timeout:
            # Clear the canvas area where the stroke was drawn
            imgCanvas = np.zeros((height, width, 3), np.uint8)
            
            # Redraw all previous corrected strokes
            for stroke_data in all_strokes:
                shape_type = process_completed_stroke(stroke_data['points'], imgCanvas, 
                                                    stroke_data['color'], stroke_data['thickness'])
            
            # Process and add the current stroke
            stroke_data = {
                'points': current_stroke.copy(),
                'color': drawColor,
                'thickness': thickness
            }
            all_strokes.append(stroke_data)
            shape_type = process_completed_stroke(current_stroke, imgCanvas, drawColor, thickness)
            
            # NEW: Update statistics
            stats['total_strokes'] += 1
            stats[shape_type] += 1
            
            # Clear undo stack when new stroke is added
            undo_stack = []
            
            current_stroke = []

        # Overlay header
        image[0:125, 0:width] = header
        
        # NEW: Draw UI elements
        draw_ui_elements(image)

        # Combine canvas with camera feed
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 5, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(image, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        cv2.imshow('Enhanced MediaPipe Drawing', img)  # NEW: Updated window title
        
        # NEW: Keyboard shortcuts
        key = cv2.waitKey(3) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):  # Save drawing
            save_drawing(all_strokes, f"drawing_{int(time.time())}.json")
        elif key == ord('u'):  # Undo
            undo_last_stroke()
        elif key == ord('r'):  # Redo
            redo_last_stroke()
        elif key == ord('c'):  # Clear
            imgCanvas = np.zeros((height, width, 3), np.uint8)
            current_stroke = []
            all_strokes = []
            undo_stack = []
            stats.update({
                'total_strokes': 0, 'lines': 0, 'circles': 0, 
                'rectangles': 0, 'freeform': 0
            })

cap.release()
cv2.destroyAllWindows()

# NEW: Print final session statistics
print("\n=== SESSION SUMMARY ===")
print(f"Total drawing time: {int(time.time() - stats['session_start'])} seconds")
print(f"Total strokes: {stats['total_strokes']}")
print(f"Lines: {stats['lines']}")
print(f"Circles: {stats['circles']}")
print(f"Rectangles: {stats['rectangles']}")
print(f"Freeform drawings: {stats['freeform']}")
print("Thank you for using Enhanced Hand Drawing!")