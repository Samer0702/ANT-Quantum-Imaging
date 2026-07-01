# ANT Quantum Imaging

This application controls a Thorlabs camera and a Thorlabs piezo stage to run
Quantum Imaging with Undetected Photons (QIUP) acquisitions and visualize the
results in real time.


> **Windows only.** This app talks to the lab hardware (Thorlabs camera and
> Thorlabs Kinesis piezo controller) through Windows-only drivers and DLLs.
> It will not run on macOS or Linux.

---

## What you need before starting

- A Windows PC (Windows 10 or 11)
- **Python 3.14** installed
  - Download it from [python.org/downloads](https://www.python.org/downloads/)
  - During installation, **tick the box that says "Add python.exe to PATH"**
    — this matters, don't skip it.
- The hardware connected and powered on (if you're running a real
  acquisition): Thorlabs CMOS camera, Thorlabs KPZ101 piezo driver, and
  (optionally) the KSG101 strain gauge reader, plus their USB drivers
  installed via Thorlabs' Kinesis software.
- This project folder, downloaded/cloned onto your computer.

---

## Step 1 — Open a terminal (cmd) in the project folder

1. Open the project folder in File Explorer.
2. Click anywhere inside the folder, right-click and select 'Open in terminal' or similar.

You should see a black window with a prompt ending in something like:
```
...\ANT-Quantum-Imaging>
```

Every command below is typed into this terminal, one at a time.

---

## Step 2 — Create a virtual environment (venv)

A virtual environment is just an isolated folder where all the Python
packages this project needs get installed, so they don't clash with anything
else on your computer.

Create it by running:

```
python -m venv .venv
```

This creates a new folder called `.venv` inside the project. You only need
to do this **once**.

## Step 3 — Activate the virtual environment

Every time you want to run or work on this project, you first need to
"activate" the venv in your terminal:

```
.venv\Scripts\activate
```

You'll know it worked because the prompt will now start with `(.venv)`,
like this:
```
(.venv) ...\ANT-Quantum-Imaging>
```


## Step 4 — Install the required packages

With the venv activated (you should see `(.venv)` in the prompt), install
everything the app needs:

```
pip install -r dependencies.txt
```

This will take a few minutes — pip is downloading and installing about 40
packages (numpy, OpenCV, PyQt5, etc.).

---

## Step 5 — Run the app

Still inside the activated venv, move into the app folder and start it:

```
cd QIUP-APP\main_open
python main_open.py
```

The application window should open. To run it again later, just repeat
**Step 3** (activate the venv) and **Step 5** (run the app) — you don't need
to repeat Steps 1, 2, or 4 again unless something goes wrong.

---

## Quick reference for next time

Every time you want to use the app after the first setup:

```
.venv\Scripts\activate
cd QIUP-APP\main_open
python main_open.py
```

## Project structure (for reference)

- `QIUP-APP/main_open/main_open.py` — main application entry point (run this one)
- `QIUP-APP/main_open/camera_control.py` — camera control + FFT-based image analysis
- `QIUP-APP/main_open/piezo_control_open.py` — piezo stage / strain gauge control in open loop
- `acquisition_workers.py` — Single and Live Acquisition workers + Raw camera feed
- `QIUP-APP/main_open/`, `QIUP-APP/main_closed/` — open-loop / closed-loop
  acquisition variants
- `depndencies.txt` — list of required Python packages