import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Import your existing hardware controllers
from piezo_control_closed import PiezoController
from camera_control import CameraController

def run_report_acquisition():
    # 1. Initialize Hardware
    piezo = PiezoController()
    camera = CameraController()

    print("Connecting to hardware...")
    p_ok, p_msg = piezo.connect()
    c_ok = camera.connect()

    if not (p_ok and c_ok):
        print(f"Hardware connection failed. Piezo: {p_msg}, Camera: {c_ok}")
        return

    camera.set_single_frame_mode()

    # 2. Acquisition Parameters
    n_frames = 60
    total_range_um = 0.810
    step_um = total_range_um / n_frames
    settling_time_s = 0.7  # 10 ms settling time (Note: 0.7s is 700ms, kept as in your original)

    # 3. ROI Setup (Center 200px)
    h = camera.image_height
    w = camera.image_width
    cy, cx = h // 2, w // 2
    
    roi_size = 100
    half_roi = roi_size // 2
    y_min, y_max = cy - half_roi, cy + half_roi
    x_min, x_max = cx - half_roi, cx + half_roi

    # 4. Spatio-Temporal Stripe Setup (50 pixels high)
    stripe_height = 50
    slice_start = half_roi - (stripe_height // 2)
    slice_end = half_roi + (stripe_height // 2)

    total_passes = 3
    print(f"Starting {total_passes}-pass sequence ({n_frames} frames per pass)...")

    # 5. Multi-Pass Scanning Loop
    for pass_idx in range(total_passes):
        is_recording = (pass_idx == 2) # Only record on the 3rd pass (index 2)
        
        pass_type = "RECORDING" if is_recording else "WARM-UP"
        print(f"\n--- Scan pass {pass_idx + 1}/{total_passes}: {pass_type} ---")

        # Initialize data structures only on the recording pass
        if is_recording:
            intensities = []
            positions = []
            spatio_temporal_img = np.zeros((stripe_height, n_frames), dtype=np.float32)

        for i in range(n_frames):
            target_um = i * step_um
            piezo.move_to_um(target_um)
            time.sleep(settling_time_s)

            # If warming up, we just let the piezo settle and move on.
            # If recording, we do the full camera acquisition and data processing.
            if is_recording:
                # Flush any old frames from the buffer
                while camera.camera.get_pending_frame_or_null() is not None:
                    pass
                
                # Trigger and grab
                camera.camera.issue_software_trigger()
                
                frame_obj = None
                while frame_obj is None:
                    frame_obj = camera.camera.get_pending_frame_or_null()
                    time.sleep(0.001)

                # Format image
                img = np.copy(frame_obj.image_buffer).reshape(h, w)
                img = cv2.flip(img, 0)
                img = cv2.blur(img, (3, 3)) # Optional: slight blur to reduce noise in mean calculation
                
                # Extract the 200x200 center ROI for the mean calculation
                roi = img[y_min:y_max, x_min:x_max]

                # Calculate mean intensity of the 200px ROI
                mean_int = float(np.mean(roi))
                intensities.append(mean_int)
                positions.append(target_um)

                # Build the time-evolution representation (Middle 50px of the center column)
                spatio_temporal_img[:, i] = roi[slice_start:slice_end, half_roi]
                
                print(f"Frame {i+1}/{n_frames} acquired at {target_um:.3f} µm")
            else:
                # Optional: print a simpler progress indicator for warm-up passes
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"Warming up... step {i+1}/{n_frames} at {target_um:.3f} µm")

        # Reset piezo to 0 before the next pass to ensure a consistent mechanical cycle
        if pass_idx < total_passes - 1:
            print("Returning piezo to 0.0 µm for next pass...")
            piezo.move_to_um(0.0)
            time.sleep(1.0) # Give it a second to fully settle back at zero

    # Clean shutdown
    piezo.disconnect()
    camera.disconnect()

    # 6. Plotting results (using the data from the 3rd pass)
    print("\nAcquisition complete. Plotting data...")
    plot_report_data(positions, intensities, spatio_temporal_img)

def plot_report_data(positions, intensities, spatio_temporal_img):
    # Theme configuration matched to main_closed.py
    bg = "#1e1e1e"
    fg = "#d4d4d4"
    border = "#3f3f46"
    img_bg = "#121212"
    accent = "#0078d4"

    # Create a 2x1 stacked layout where the top graph is much taller than the bottom stripe
    fig, (ax_plot, ax_img) = plt.subplots(
        2, 1, 
        figsize=(10, 6), 
        gridspec_kw={'height_ratios': [5, 1]}, # 5:1 ratio gives the "thin stripe" look
        sharex=True                            # Lock X-axes together
    )
    fig.patch.set_facecolor(bg)

    # --- Top Plot: Intensity vs Position ---
    ax_plot.set_facecolor(img_bg)
    ax_plot.grid(True, color="#2d2d30", linestyle="--", linewidth=0.5, zorder=0)
    ax_plot.plot(
        positions, intensities,
        color=accent,
        linestyle="-",
        linewidth=2,
        marker="o",
        markersize=6,
        markeredgecolor="white",
        markeredgewidth=1,
        zorder=2
    )
    ax_plot.set_title("Center ROI Mean Intensity vs Piezo Position", color=fg, fontsize=12, fontweight="bold")
    ax_plot.set_ylabel("ROI Mean Intensity\n(200x200 px)", fontsize=10, color=fg, fontweight="bold")
    ax_plot.tick_params(colors=fg, labelsize=9)
    for spine in ax_plot.spines.values():
        spine.set_color(border)
    
    # Hide the x-axis tick labels for the top plot so they don't overlap the stripe
    plt.setp(ax_plot.get_xticklabels(), visible=False)

    # --- Bottom Plot: 50px Spatio-Temporal Stripe ---
    ax_img.set_facecolor(img_bg)
    norm_img = cv2.normalize(spatio_temporal_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Calculate exact extent so the pixels line up directly under the graph points
    step = positions[1] - positions[0] if len(positions) > 1 else 1
    extent = [positions[0] - step/2, positions[-1] + step/2, 50, 0]
    
    ax_img.imshow(norm_img, cmap='gray', aspect='auto', extent=extent)
    
    ax_img.set_xlabel("Piezo Position (µm)", fontsize=10, color=fg, fontweight="bold")
    ax_img.set_ylabel("50px Slice", fontsize=10, color=fg, fontweight="bold")
    ax_img.tick_params(colors=fg, labelsize=9)
    ax_img.set_yticks([]) # Hide the Y-axis numbers on the image stripe to keep it clean

    for spine in ax_img.spines.values():
        spine.set_color(border)

    # Snug the two plots closely together
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.05) 
    
    # Save the output
    plt.savefig("acquisition_report_stripe.png", facecolor=bg, dpi=300)
    print("Saved plot to 'acquisition_report_stripe.png'")
    plt.show()

if __name__ == "__main__":
    run_report_acquisition()