# 🎯 Data Scrap Tool & Advanced Dataset Viewer

[![Tkinter](https://img.shields.io/badge/Tkinter-FF6C37.svg?style=flat\&logo=python)](https://docs.python.org/3/library/tkinter.html)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=flat\&logo=streamlit)](https://streamlit.io/)
[![NumPy](https://img.shields.io/badge/NumPy-013243.svg?style=flat\&logo=numpy)](https://numpy.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-27338e.svg?style=flat\&logo=opencv)](https://opencv.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458.svg?style=flat\&logo=pandas)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75.svg?style=flat\&logo=plotly)](https://plotly.com/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C.svg?style=flat\&logo=matplotlib)](https://matplotlib.org/)

## 🚀 Overview

This comprehensive data collection and analysis suite is designed for capturing screen interactions, keyboard events, and synchronized video recordings. It facilitates detailed dataset creation suitable for machine learning and AI projects, featuring interactive GUI tools (Tkinter) and advanced analytics dashboards (Streamlit).

---

## 🛠️ Features

### 🖥️ Screen Capture & Monitoring

* **Multi-monitor support:** Automatic detection and selection.
* **Custom Region Selection:** Mouse-drag region selection for precise data capture.

### ⌨️ Keyboard Tracking

* **Dynamic Key Tracking:** Customizable key management (add/remove).
* **Multi-hot encoding:** Captures active keys as binary vectors.
* **Phase Tracking:** Detailed press/release event recording.

### 🎬 Real-Time Video Recording

* **Live Preview:** Immediate visual feedback of captures.
* **Overlay Indicators:** Real-time visual keypress indicators.
* **Synced Video Capture:** Real-time MP4 video recording synchronized with dataset snapshots.

### 📁 Dataset Management

* **Snapshot Metadata:**

  * Screen images (normalized RGB)
  * Timing intervals between key events
  * Multi-hot key vectors
  * Key hold durations
  * Press/release phase
  * Video frame synchronization
* **Export & Save:** `.npz` format datasets including metadata.

### 📊 Advanced Dataset Viewer (Tkinter GUI)

* **Interactive Data Exploration:** Navigate through captured entries.
* **Advanced Filtering:** Filter by keys, phases, and empty entries.
* **Integrated Video Player:** Analyze video in sync with dataset.
* **Metadata Inspection:** Detailed insights on each entry.

### 🔍 Comprehensive Analytics (Streamlit Dashboard)

* **Data Structure Analysis:** Insights into entry components.
* **Feature Extraction:** Derived metrics (complexity, entropy, durations).
* **Interactive Visualizations:**

  * **Phase Distribution:** Pie charts showing the distribution of phases (press/release).
  * **Key Usage Distribution:** Bar charts visualizing how frequently each key is pressed.
  * **Timeline Analysis:** Line charts showing changes in hold duration, active keys, image intensity, and complexity over time.
  * **Key Combination Analysis:** Identify and visualize the most frequent key combinations.
  * **Correlation Matrix:** Heatmaps illustrating correlations between numeric features.
  * **3D Feature Space:** Interactive 3D scatter plots displaying relationships among duration, intensity, and entropy.
* **Detailed Statistical Analysis:** Summary statistics, distributions, boxplots, histograms, and detailed comparisons.

### ▶️ Video Playback

* **Integrated Playback:** Controls (play, pause, seek, fast-forward/rewind).
* **Metadata Sync:** Frame-accurate synchronization with dataset entries.

---

## 📸 Screenshots

*Coming Soon: Screenshots showcasing application features and analytics visualizations.*

---

## 🐞 Troubleshooting

### Known Issues & Solutions

* **Issue:** Streamlit server did not start.

  * **Solution:** Try reopen the Streamlit using the dedicated button in the GUI.
  * 
* **Issue:** video does not change automatically when I change dataset.

  * **Solution:** Press the Play rewiev Video button twice to change the video.

If problems persist, please open an issue on GitHub.

---

## 📂 Project Structure

```
project_root/
├── Data_scrap_tool.py
├── dataset_viewer/
│   ├── dataset_viewer.py
│   ├── video_player.py
│   ├── gui_utils.py
│   ├── io_utils.py
│   ├── constants.py
├── Data_scrap_tool_function/
│   ├── capture.py
│   ├── data_collector.py
│   ├── overlay.py
|   ├── streamlit_app.py
|
├── collected/
│   └── datasets (*.npz)
├── recordings/
│   └── videos (*.mp4)
```

---

## 💻 Getting Started

### 📌 Installation

```bash
git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

### 🚀 Running Applications

* **Data Collection Tool:**

  ```bash
  python Data_scrap_tool.py
  ```
---


## 🧑‍💻 Contributing

Feel free to fork this project, create pull requests, or open issues!

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
