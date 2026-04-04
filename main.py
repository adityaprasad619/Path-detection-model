import cv2
import numpy as np

# Load image
img = cv2.imread("voronoi.jpg")

# Convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define range for orange/red points (tune if needed)
lower_red = np.array([5, 100, 100])
upper_red = np.array([25, 255, 255])
mask = cv2.inRange(hsv, lower_red, upper_red)

# Find contours of red points
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get image shape
h, w, _ = img.shape

# Axis range from image (graph scale is 1–12 both axes)
x_min, x_max = 1, 12
y_min, y_max = 1, 12

coordinates = []

for cnt in contours:
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])  # pixel X
        cy = int(M["m01"] / M["m00"])  # pixel Y

        # Convert pixel position to graph scale
        x_coord = x_min + (cx / w) * (x_max - x_min)
        y_coord = y_max - (cy / h) * (y_max - y_min)  # flip Y-axis since image origin is top-left

        coordinates.append((round(x_coord, 2), round(y_coord, 2)))

# Sort for readability
coordinates = sorted(coordinates, key=lambda p: (p[1], p[0]))

# Display coordinates
print("Extracted Coordinates (x, y):")
for point in coordinates:
    print(point)
