(function () {
  'use strict';

  // ─── Constants ───────────────────────────────────────────────────────────────
  const GRID_SIZE   = 121;          // -60 … +60 inclusive
  const HEIGHT_MIN  = -7;
  const HEIGHT_MAX  = 35;
  const MAX_DIM     = 800;          // downsample input if larger
  const INPAINT_R   = 20;           // inpaint blur radius
  const GREEN_BLUR_R = 3;           // green-channel box-blur radius
  const GAUSS_SIGMA  = 2;           // final smoothing sigma

  let currentStem = 'topography';

  // ─── RGB → HSV (OpenCV-style: H∈[0,180], S∈[0,255], V∈[0,255]) ─────────────
  function rgbToHsv(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min;
    let h = 0;
    if (d !== 0) {
      if      (max === r) h = ((g - b) / d) % 6;
      else if (max === g) h = (b - r) / d + 2;
      else                h = (r - g) / d + 4;
      h = h * 30;           // *180/6
      if (h < 0) h += 180;
    }
    const s = max === 0 ? 0 : d / max * 255;
    const v = max * 255;
    return [h, s, v];
  }

  // ─── Build red / yellow marker mask ──────────────────────────────────────────
  function buildMask(pixels, w, h) {
    const mask = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) {
      const r = pixels[i * 4], g = pixels[i * 4 + 1], b = pixels[i * 4 + 2];
      const [hv, s, v] = rgbToHsv(r, g, b);
      const red    = (hv <= 10 || hv >= 160) && s >= 100 && v >= 100;
      const yellow = hv >= 20 && hv <= 35    && s >= 100 && v >= 100;
      mask[i] = (red || yellow) ? 1 : 0;
    }
    return mask;
  }

  // ─── Dilate mask (two-pass separable square kernel) ──────────────────────────
  function dilateMask(mask, w, h, r) {
    const tmp = new Uint8Array(w * h);
    const out = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let found = false;
        for (let dx = -r; dx <= r && !found; dx++) {
          const nx = x + dx;
          if (nx >= 0 && nx < w && mask[y * w + nx]) found = true;
        }
        tmp[y * w + x] = found ? 1 : 0;
      }
    }
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let found = false;
        for (let dy = -r; dy <= r && !found; dy++) {
          const ny = y + dy;
          if (ny >= 0 && ny < h && tmp[ny * w + x]) found = true;
        }
        out[y * w + x] = found ? 1 : 0;
      }
    }
    return out;
  }

  // ─── Summed-area table box blur (O(n), handles any radius) ───────────────────
  function boxBlurIntegral(data, w, h, r) {
    const W = w + 1;
    const integral = new Float64Array(W * (h + 1));
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        integral[(y + 1) * W + (x + 1)] =
          data[y * w + x]
          + integral[y * W + (x + 1)]
          + integral[(y + 1) * W + x]
          - integral[y * W + x];
      }
    }
    const result = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const x1 = Math.max(0, x - r), y1 = Math.max(0, y - r);
        const x2 = Math.min(w - 1, x + r), y2 = Math.min(h - 1, y + r);
        const area = (x2 - x1 + 1) * (y2 - y1 + 1);
        const sum  = integral[(y2 + 1) * W + (x2 + 1)]
                   - integral[y1 * W + (x2 + 1)]
                   - integral[(y2 + 1) * W + x1]
                   + integral[y1 * W + x1];
        result[y * w + x] = sum / area;
      }
    }
    return result;
  }

  // ─── Inpaint: normalized convolution (fills masked pixels from neighbours) ───
  function inpaint(pixels, mask, w, h) {
    const result = new Uint8ClampedArray(pixels);
    for (let c = 0; c < 3; c++) {
      const vals = new Float32Array(w * h);
      const wts  = new Float32Array(w * h);
      for (let i = 0; i < w * h; i++) {
        if (!mask[i]) { vals[i] = pixels[i * 4 + c]; wts[i] = 1; }
      }
      const bv = boxBlurIntegral(vals, w, h, INPAINT_R);
      const bw = boxBlurIntegral(wts,  w, h, INPAINT_R);
      for (let i = 0; i < w * h; i++) {
        if (mask[i]) {
          result[i * 4 + c] = bw[i] > 0.001 ? Math.round(bv[i] / bw[i]) : 128;
        }
      }
    }
    return result;
  }

  // ─── Build height grid (GRID_SIZE × GRID_SIZE) ───────────────────────────────
  function buildHeightGrid(green, w, h) {
    const gridSum = new Float64Array(GRID_SIZE * GRID_SIZE);
    const gridCnt = new Int32Array(GRID_SIZE * GRID_SIZE);

    for (let py = 0; py < h; py++) {
      const sy = Math.round(60 - (py / h) * 120);          // flipped y
      const gy = sy + 60;
      if (gy < 0 || gy >= GRID_SIZE) continue;

      for (let px = 0; px < w; px++) {
        const sx = Math.round((px / w) * 120 - 60);
        const gx = sx + 60;
        if (gx < 0 || gx >= GRID_SIZE) continue;

        const gv     = green[py * w + px];
        const height = ((255 - gv) / 255) * (HEIGHT_MAX - HEIGHT_MIN) + HEIGHT_MIN;
        const gi     = gy * GRID_SIZE + gx;
        gridSum[gi] += height;
        gridCnt[gi]++;
      }
    }

    const grid = new Float32Array(GRID_SIZE * GRID_SIZE).fill(NaN);
    for (let i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
      if (gridCnt[i] > 0) grid[i] = gridSum[i] / gridCnt[i];
    }
    return grid;
  }

  // ─── IQR outlier filter ───────────────────────────────────────────────────────
  function filterIQR(grid) {
    const vals = Array.from(grid).filter(v => !isNaN(v)).sort((a, b) => a - b);
    const q1   = vals[Math.floor(vals.length * 0.25)];
    const q3   = vals[Math.floor(vals.length * 0.75)];
    const iqr  = q3 - q1;
    const lo   = q1 - 1.5 * iqr;
    const hi   = q3 + 1.5 * iqr;
    const result = new Float32Array(grid);
    for (let i = 0; i < result.length; i++) {
      if (!isNaN(result[i]) && (result[i] < lo || result[i] > hi)) result[i] = NaN;
    }
    return result;
  }

  // ─── Fill NaN cells by iterative neighbour averaging ─────────────────────────
  function fillNaN(grid) {
    const result = new Float32Array(grid);
    const dirs   = [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]];
    for (let iter = 0; iter < 30; iter++) {
      let changed = false;
      for (let y = 0; y < GRID_SIZE; y++) {
        for (let x = 0; x < GRID_SIZE; x++) {
          if (!isNaN(result[y * GRID_SIZE + x])) continue;
          let sum = 0, cnt = 0;
          for (const [dy, dx] of dirs) {
            const nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE) {
              const v = result[ny * GRID_SIZE + nx];
              if (!isNaN(v)) { sum += v; cnt++; }
            }
          }
          if (cnt > 0) { result[y * GRID_SIZE + x] = sum / cnt; changed = true; }
        }
      }
      if (!changed) break;
    }
    return result;
  }

  // ─── Separable Gaussian smooth ────────────────────────────────────────────────
  function gaussianKernel(sigma) {
    const r = Math.ceil(3 * sigma);
    const k = [];
    let sum = 0;
    for (let i = -r; i <= r; i++) {
      const v = Math.exp(-(i * i) / (2 * sigma * sigma));
      k.push(v); sum += v;
    }
    return k.map(v => v / sum);
  }

  function convolveH(data, kernel) {
    const r      = (kernel.length - 1) >> 1;
    const result = new Float32Array(data.length);
    for (let y = 0; y < GRID_SIZE; y++) {
      for (let x = 0; x < GRID_SIZE; x++) {
        let sum = 0, wsum = 0;
        for (let k = 0; k < kernel.length; k++) {
          const nx = x + k - r;
          if (nx >= 0 && nx < GRID_SIZE) {
            const v = data[y * GRID_SIZE + nx];
            if (!isNaN(v)) { sum += kernel[k] * v; wsum += kernel[k]; }
          }
        }
        result[y * GRID_SIZE + x] = wsum > 0 ? sum / wsum : NaN;
      }
    }
    return result;
  }

  function convolveV(data, kernel) {
    const r      = (kernel.length - 1) >> 1;
    const result = new Float32Array(data.length);
    for (let y = 0; y < GRID_SIZE; y++) {
      for (let x = 0; x < GRID_SIZE; x++) {
        let sum = 0, wsum = 0;
        for (let k = 0; k < kernel.length; k++) {
          const ny = y + k - r;
          if (ny >= 0 && ny < GRID_SIZE) {
            const v = data[ny * GRID_SIZE + x];
            if (!isNaN(v)) { sum += kernel[k] * v; wsum += kernel[k]; }
          }
        }
        result[y * GRID_SIZE + x] = wsum > 0 ? sum / wsum : NaN;
      }
    }
    return result;
  }

  function gaussianSmooth(grid, sigma) {
    const k = gaussianKernel(sigma);
    return convolveV(convolveH(grid, k), k);
  }

  // ─── Main pipeline ────────────────────────────────────────────────────────────
  async function processImage(file) {
    currentStem = file.name.replace(/\.[^.]+$/, '');
    setStatus('Loading image…');

    const bitmap = await createImageBitmap(file);
    let { width: w, height: h } = bitmap;

    if (w > MAX_DIM || h > MAX_DIM) {
      const scale = MAX_DIM / Math.max(w, h);
      w = Math.round(w * scale);
      h = Math.round(h * scale);
    }

    const canvas = new OffscreenCanvas(w, h);
    const ctx    = canvas.getContext('2d');
    ctx.drawImage(bitmap, 0, 0, w, h);
    const pixels = ctx.getImageData(0, 0, w, h).data;

    setStatus('Detecting markers…');
    await tick();
    const rawMask = buildMask(pixels, w, h);
    const mask    = dilateMask(rawMask, w, h, 7);

    setStatus('Inpainting markers…');
    await tick();
    const inpainted = inpaint(pixels, mask, w, h);

    setStatus('Extracting height data…');
    await tick();

    // Extract and smooth green channel
    const green = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) green[i] = inpainted[i * 4 + 1];
    const blurred = boxBlurIntegral(green, w, h, GREEN_BLUR_R);

    // Build grid
    const rawGrid      = buildHeightGrid(blurred, w, h);
    const filteredGrid = filterIQR(rawGrid);
    const filledGrid   = fillNaN(filteredGrid);
    const smoothGrid   = gaussianSmooth(filledGrid, GAUSS_SIGMA);

    setStatus('Rendering 3D plot…');
    await tick();

    // Build 2D z array: z[gy][gx], gy=0 → y=-60, gy=120 → y=60
    const z = [];
    for (let gy = 0; gy < GRID_SIZE; gy++) {
      const row = [];
      for (let gx = 0; gx < GRID_SIZE; gx++) {
        row.push(smoothGrid[gy * GRID_SIZE + gx]);
      }
      z.push(row);
    }

    // Build CSV rows [x, y, rounded_height]
    const csvRows = [['x', 'y', 'height']];
    for (let gy = 0; gy < GRID_SIZE; gy++) {
      for (let gx = 0; gx < GRID_SIZE; gx++) {
        const hv = smoothGrid[gy * GRID_SIZE + gx];
        if (!isNaN(hv)) {
          csvRows.push([gx - 60, gy - 60, Math.round(hv)]);
        }
      }
    }

    renderPlot(z, csvRows);
    setStatus('');
  }

  // ─── Render Plotly surface ────────────────────────────────────────────────────
  function renderPlot(z, csvRows) {
    const axis = Array.from({ length: GRID_SIZE }, (_, i) => i - 60);
    const plotDiv = document.getElementById('plot');

    Plotly.newPlot(plotDiv, [{
      type:       'surface',
      x:          axis,
      y:          axis,
      z:          z,
      colorscale: 'Earth',
      showscale:  true,
      colorbar: {
        title:     { text: 'Height', side: 'right', font: { color: '#333', size: 12 } },
        tickfont:  { color: '#616161', size: 11 },
        thickness: 14,
        bgcolor:   '#fafafa',
        bordercolor: '#e0e0e0',
      },
    }], {
      paper_bgcolor: '#fafafa',
      plot_bgcolor:  '#fafafa',
      scene: {
        bgcolor: '#fafafa',
        xaxis: { title: '', showticklabels: false, showgrid: true, gridcolor: '#e8e8e8', zeroline: false },
        yaxis: { title: '', showticklabels: false, showgrid: true, gridcolor: '#e8e8e8', zeroline: false },
        zaxis: { title: 'Height', tickfont: { color: '#616161', size: 11 }, color: '#333' },
        camera:      { eye: { x: 1.8, y: -1.8, z: 1.0 } },
        aspectratio: { x: 2, y: 2, z: 0.6 },
      },
      margin: { t: 48, b: 0, l: 0, r: 80 },
      title:  {
        text: `3D Topography — ${currentStem}`,
        font: { color: '#1e1e1e', size: 15, family: "'Segoe UI', system-ui, sans-serif" },
        x: 0.04, xanchor: 'left',
      },
    }, { responsive: true });

    window._csvRows = csvRows;
    document.getElementById('result-label').textContent = currentStem;
    document.getElementById('results').classList.remove('hidden');
    document.getElementById('drop-section').classList.add('hidden');

    // Enable sidebar export links
    ['nav-png', 'nav-csv', 'nav-xlsx'].forEach(id => {
      document.getElementById(id).classList.add('enabled');
      document.getElementById(id).style.removeProperty('pointer-events');
      document.getElementById(id).style.removeProperty('opacity');
    });
    document.getElementById('btn-reset').classList.remove('hidden');
  }

  // ─── Downloads ────────────────────────────────────────────────────────────────
  function downloadCSV() {
    const csv  = window._csvRows.map(r => r.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    triggerDownload(URL.createObjectURL(blob), `${currentStem}_height_data.csv`, true);
  }

  function downloadXLSX() {
    const ws = XLSX.utils.aoa_to_sheet(window._csvRows);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, 'Height Data');
    XLSX.writeFile(wb, `${currentStem}_height_data.xlsx`);
  }

  async function downloadPNG() {
    const url = await Plotly.toImage(
      document.getElementById('plot'),
      { format: 'png', width: 1600, height: 1100 }
    );
    triggerDownload(url, `${currentStem}_3d_topo.png`, false);
  }

  function triggerDownload(href, filename, revoke) {
    const a = document.createElement('a');
    a.href = href;
    a.download = filename;
    a.click();
    if (revoke) setTimeout(() => URL.revokeObjectURL(href), 1000);
  }

  // ─── Helpers ─────────────────────────────────────────────────────────────────
  function setStatus(msg) {
    const bar  = document.getElementById('status');
    const text = document.getElementById('status-text');
    text.textContent = msg;
    bar.classList.toggle('hidden', !msg);
  }

  function tick() {
    return new Promise(r => setTimeout(r, 0));
  }

  // ─── UI wiring ────────────────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', () => {
    const dropZone  = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');

    dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', ()  => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('image/')) processImage(file);
    });
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', () => {
      if (fileInput.files[0]) processImage(fileInput.files[0]);
    });

    document.getElementById('nav-csv').addEventListener('click',  e => { e.preventDefault(); if (window._csvRows) downloadCSV(); });
    document.getElementById('nav-xlsx').addEventListener('click', e => { e.preventDefault(); if (window._csvRows) downloadXLSX(); });
    document.getElementById('nav-png').addEventListener('click',  e => { e.preventDefault(); if (window._csvRows) downloadPNG(); });
    document.getElementById('btn-reset').addEventListener('click', () => {
      Plotly.purge(document.getElementById('plot'));
      document.getElementById('results').classList.add('hidden');
      document.getElementById('drop-section').classList.remove('hidden');
      document.getElementById('btn-reset').classList.add('hidden');
      window._csvRows = null;
      ['nav-png', 'nav-csv', 'nav-xlsx'].forEach(id => {
        const el = document.getElementById(id);
        el.classList.remove('enabled');
        el.style.pointerEvents = 'none';
        el.style.opacity = '0.4';
      });
    });
  });
})();
