import shutil
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from pathlib import Path

# -------- Paths -------- #
SRC_DIR = Path("src_pic")
OUT_DIR = Path("result_pic")

# Clear and recreate the output folder on each run
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir()

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

image_files = [f for f in SRC_DIR.iterdir() if f.suffix.lower() in SUPPORTED_EXTS]

if not image_files:
    raise FileNotFoundError(f"No supported images found in '{SRC_DIR}/'")

print(f"Found {len(image_files)} image(s) to process.")


def process_image(img_path: Path):
    stem = img_path.stem
    print(f"\nProcessing: {img_path.name}")

    img = cv.imread(str(img_path))
    if img is None:
        print(f"  WARNING: Could not load '{img_path.name}', skipping.")
        return

    # -------- Mask out the red block -------- #
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask_red1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask_red1 | mask_red2

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv.inRange(hsv, lower_yellow, upper_yellow)

    marker_mask = red_mask | yellow_mask
    marker_mask = cv.dilate(marker_mask, np.ones((15, 15), np.uint8), iterations=2)

    img_clean = cv.inpaint(img, marker_mask, inpaintRadius=20, flags=cv.INPAINT_TELEA)

    # -------- Extract green channel & denoise -------- #
    g = cv.split(img_clean)[1]
    g = cv.medianBlur(g, 7)
    g = cv.bilateralFilter(g, 9, 75, 75)

    h_px, w_px = img.shape[:2]

    # -------- Build height data -------- #
    def green_to_height(green_value):
        return round(((255 - green_value) / 255) * (35 - (-7)) + (-7))

    def scale_x(value):
        return round((value / w_px) * 120 - 60)

    def scale_y_flipped(value):
        scaled = (value / h_px) * 120 - 60
        return round(60 - (scaled + 60))

    ys, xs = np.mgrid[0:h_px, 0:w_px]
    heights = np.vectorize(green_to_height)(g)
    scaled_xs = np.vectorize(scale_x)(xs)
    scaled_ys = np.vectorize(scale_y_flipped)(ys)

    data = np.stack([scaled_xs.ravel(), scaled_ys.ravel(), heights.ravel()], axis=1)
    df = pd.DataFrame(data, columns=["x", "y", "height"])

    # -------- Remove outliers (IQR) -------- #
    Q1 = df["height"].quantile(0.25)
    Q3 = df["height"].quantile(0.75)
    IQR = Q3 - Q1
    df_filtered = df[(df["height"] > Q1 - 1.5 * IQR) & (df["height"] < Q3 + 1.5 * IQR)]

    df_avg = df_filtered.groupby(["x", "y"]).agg({"height": "mean"}).reset_index()
    df_avg["height"] = df_avg["height"].round().astype(int)

    # -------- Save CSV -------- #
    csv_path = OUT_DIR / f"{stem}_height_data.csv"
    df_avg.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")

    # -------- Save Excel -------- #
    xlsx_path = OUT_DIR / f"{stem}_height_data.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_avg.to_excel(writer, index=False, sheet_name="Height Data")
    print(f"  Saved Excel: {xlsx_path}")

    # -------- 3D Surface Plot -------- #
    grid_x, grid_y = np.mgrid[-60:61, -60:61]
    points = df_avg[["x", "y"]].values
    values = df_avg["height"].values

    grid_z = griddata(points, values, (grid_x, grid_y), method="linear")
    grid_z = gaussian_filter(grid_z, sigma=2)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap="terrain", edgecolor="none",
                           linewidth=0, antialiased=True)

    ax.set_title(f"3D Topography — {stem}", fontsize=18, fontweight="bold")

    ax.set_xlim([-60, 60])
    ax.set_ylim([-60, 60])
    z_min = np.nanmin(grid_z)
    z_max = np.nanmax(grid_z)
    z_padding = (z_max - z_min) * 0.1
    ax.set_zlim([z_min - z_padding, z_max + z_padding])

    ax.set_box_aspect([3, 3, 1])
    ax.view_init(elev=45, azim=-50)
    ax.dist = 7

    ax.grid(False)
    ax.set_axis_off()

    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.02)
    cbar.set_label("Height", fontsize=14, fontweight="bold")
    cbar.ax.tick_params(labelsize=11)

    plt.subplots_adjust(left=0.0, right=1.0, top=0.95, bottom=0.0)
    plt.tight_layout()

    plot_path = OUT_DIR / f"{stem}_3d_topo.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 3D plot: {plot_path}")


for img_file in image_files:
    process_image(img_file)

print("\nAll done.")
