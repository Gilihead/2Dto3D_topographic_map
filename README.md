# 2D → 3D Topographic Map

Extracts height data from a coloured 2D topographic map image and generates an interactive 3D surface, CSV, and Excel output.

## Two ways to use

### Python script (batch)
Reads all images in `src_pic/`, writes 3D plots and height data to `result_pic/`.
```bash
pip install numpy scipy matplotlib opencv-python openpyxl
python apsc_101_topo_3.py
```

### Browser extension (Chrome / Edge)
Drop one image in, download the 3D plot (PNG), CSV, and Excel instantly.
```bash
python package_extension.py   # builds topo3d_extension.zip
```
Then load the zip following `extension/INSTALL.txt`.

## Output
| File | Description |
|------|-------------|
| `*_3d_topo.png` | Rendered 3D surface plot |
| `*_height_data.csv` | x, y, height table |
| `*_height_data.xlsx` | Same data as Excel workbook |
