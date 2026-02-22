import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


# Load the image
img = cv.imread('D:/myCodeP/LearnOpenCV/Photos/topography.jpg')

if img is None:
    raise FileNotFoundError("Error: Unable to load image")

# -------- Mask out the red block -------- #
# Convert to HSV to detect the red square
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Red in HSV wraps around 0/180, so we need two ranges
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

mask_red1 = cv.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv.inRange(hsv, lower_red2, upper_red2)
red_mask = mask_red1 | mask_red2

# Also detect the yellow dot near the red block
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([35, 255, 255])
yellow_mask = cv.inRange(hsv, lower_yellow, upper_yellow)

# Combine masks and dilate to cover edges of the markers
marker_mask = red_mask | yellow_mask
marker_mask = cv.dilate(marker_mask, np.ones((15, 15), np.uint8), iterations=2)

# Inpaint the marker regions so they blend with surrounding topography
img_clean = cv.inpaint(img, marker_mask, inpaintRadius=20, flags=cv.INPAINT_TELEA)

# Extract the green channel from the cleaned image and apply noise reduction
g = cv.split(img_clean)[1]  # Green channel only
g = cv.medianBlur(g, 7)
g = cv.bilateralFilter(g, 9, 75, 75)

# Convert green value to height (light green = low, dark green = high)
def green_to_height(green_value):
    max_green = 255
    height = ((max_green - green_value) / max_green) * (35 - (-7)) + (-7)
    return round(height)

# Scale x to range -60 to 60
def scale_x(value, original_max):
    return round(((value / original_max) * 120) - 60)

# Scale and flip y to range -60 to 60
def scale_y_flipped(value, original_max):
    scaled = ((value / original_max) * 120) - 60
    flipped = 60 - (scaled + 60)
    return round(flipped)

# Extract (x, y, height) data
data = []

for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        green_value = g[y, x]
        height = green_to_height(green_value)
        scaled_x = scale_x(x, original_max=img.shape[1])
        scaled_y = scale_y_flipped(y, original_max=img.shape[0])
        data.append((scaled_x, scaled_y, height))

# Data to DataFrame
df = pd.DataFrame(data, columns=["x", "y", "height"])

# -------- Remove Outliers Using Interquartile Range (IQR) -------- #
Q1 = df["height"].quantile(0.25)
Q3 = df["height"].quantile(0.75)
IQR = Q3 - Q1

df_filtered = df[(df["height"] > Q1 - 1.5 * IQR) & (df["height"] < Q3 + 1.5 * IQR)]

# Average duplicate (x, y) and round height
df_avg = df_filtered.groupby(["x", "y"]).agg({"height": "mean"}).reset_index()
df_avg["height"] = df_avg["height"].round().astype(int)

# Save to CSV (uncomment if needed)
# df_avg.to_csv("filtered_height_data.csv", index=False)
print("Data processed successfully")

# -------- 3D Smooth Surface Plot -------- #

# Create grid for surface
grid_x, grid_y = np.mgrid[-60:61, -60:61]
points = df_avg[["x", "y"]].values
values = df_avg["height"].values

# Interpolate height values onto the grid
grid_z = griddata(points, values, (grid_x, grid_y), method='linear')

# Apply Gaussian smoothing to the surface
grid_z = gaussian_filter(grid_z, sigma=2)

# Plot the surface
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='terrain', edgecolor='none', linewidth=0, antialiased=True)

ax.set_title('Denoised 3D Topography', fontsize=20, fontweight='bold')

# -------- FIX: Widen the view so the surface doesn't clip -------- #
ax.set_xlim([-60, 60])
ax.set_ylim([-60, 60])

z_min = np.nanmin(grid_z)
z_max = np.nanmax(grid_z)
z_padding = (z_max - z_min) * 0.1
ax.set_zlim([z_min - z_padding, z_max + z_padding])

# Stretch x and y relative to z so the model looks flattened
# x and y span 120 units, z spans ~30-40, so we scale accordingly
ax.set_box_aspect([3, 3, 1])  # (x, y, z) ratio — makes horizontal much wider than vertical

# Adjust the camera: higher elevation + further distance to see more
ax.view_init(elev=45, azim=-50)
ax.dist = 7

ax.grid(False)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_zlabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_axis_off()

# Add a vertical colorbar showing height
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.02)
cbar.set_label('Height', fontsize=14, fontweight='bold')
cbar.ax.tick_params(labelsize=11)

plt.subplots_adjust(left=0.0, right=1.0, top=0.95, bottom=0.0)
plt.tight_layout()
plt.show()