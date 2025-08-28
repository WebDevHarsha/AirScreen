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
undo_stack = []
stroke_timeout = 30  # frames without drawing before considering stroke complete

# Colors and features
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

color_names = ["Red", "Blue", "Green", "Black", "Cyan", "Magenta", "Yellow", "Purple"]

# Gesture detection variables
gesture_start_time = 0
gesture_threshold = 1.0  # seconds to hold gesture
show_gesture_feedback = False
gesture_feedback_text = ""

# Drawing variables
drawColor = (0, 0, 255)
thickness = 20 
tipIds = [4, 8, 12, 16, 20] 
xp, yp = [0, 0] 
frame_count = 0
last_draw_frame = 0
color_index = 0

# Statistics tracking
stats = {
    'total_strokes': 0,
    'line': 0,        # Changed from 'lines' to match function return
    'circle': 0,      # Changed from 'circles' to match function return  
    'rectangle': 0,   # Changed from 'rectangles' to match function return
    'freeform': 0,
    'session_start': time.time()
}

def create_navbar(image):
    """Create a custom navbar using Python drawing functions"""
    # Create navbar background
    navbar_height = 100
    cv2.rectangle(image, (0, 0), (width, navbar_height), (50, 50, 50), -1)
    
    # Color palette
    color_width = 120
    color_height = 60
    start_x = 50
    start_y = 20
    
    for i, (color, name) in enumerate(zip(colors, color_names)):
        x1 = start_x + i * color_width
        y1 = start_y
        x2 = x1 + color_width - 10
        y2 = y1 + color_height
        
        # Draw color rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
        
        # Add white border
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        # Highlight current selection
        if i == color_index:
            cv2.rectangle(image, (x1-3, y1-3), (x2+3, y2+3), (255, 255, 255), 4)
        
        # Add color name
        text_color = (255, 255, 255) if color != (0, 255, 255) else (0, 0, 0)  # White text, black for yellow
        cv2.putText(image, name, (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
    
    # Add title
    cv2.putText(image, "Enhanced Hand Drawing", (width//2 - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add separator line
    cv2.line(image, (0, navbar_height), (width, navbar_height), (255, 255, 255), 2)

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

def draw_ui_elements(image):
    """Draw UI elements like current color and thickness indicators"""
    global show_gesture_feedback, gesture_feedback_text
    
    # Display current color indicator
    cv2.circle(image, (50, 150), 20, drawColor, -1)
    cv2.circle(image, (50, 150), 20, (255, 255, 255), 2)
    cv2.putText(image, "Color", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Display thickness indicator
    cv2.circle(image, (50, 220), thickness, drawColor, 2)
    cv2.putText(image, "Thickness", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
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
        cv2.putText(image, text, (10, height - 160 + i * 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

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

                    # Check for special gestures first
                    if detect_special_gestures(fingers, current_time):
                        continue

                    # Selection mode
                    nonSel = [0, 3, 4] 
                    if (fingers[1] and fingers[2]) and all(fingers[i] == 0 for i in nonSel):
                        xp, yp = [x1, y1]

                        if y1 < 100:  # Updated for new navbar height
                            # Color selection based on navbar layout
                            color_width = 120
                            start_x = 50
                            
                            for i in range(len(colors)):
                                section_start = start_x + i * color_width
                                section_end = section_start + color_width - 10
                                
                                if section_start < x1 < section_end:
                                    drawColor = colors[i]
                                    color_index = i
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
                        undo_stack = []
                        stats.update({
                            'total_strokes': 0, 'line': 0, 'circle': 0, 
                            'rectangle': 0, 'freeform': 0
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
                        if mod_v > 0:
                            v1, v2 = [v1/mod_v, v2/mod_v]
                        
                        c = 3 + r
                        x0, y0 = [int(x0 - v1*c), int(y0 - v2*c)]
                        cv2.circle(image, (x0, y0), int(r/2), drawColor, -1)

                        if fingers[4]:                        
                            thickness = max(5, min(50, r))
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
            
            # Update statistics
            stats['total_strokes'] += 1
            stats[shape_type] += 1
            
            # Clear undo stack when new stroke is added
            undo_stack = []
            
            current_stroke = []

        # Create custom navbar
        create_navbar(image)
        
        # Draw UI elements
        draw_ui_elements(image)

        # Combine canvas with camera feed
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 5, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(image, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        cv2.imshow('Enhanced Hand Drawing', img)
        
        # Keyboard shortcuts
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
                'total_strokes': 0, 'line': 0, 'circle': 0, 
                'rectangle': 0, 'freeform': 0
            })

cap.release()
cv2.destroyAllWindows()

print("\nThank you for using Enhanced Hand Drawing!")