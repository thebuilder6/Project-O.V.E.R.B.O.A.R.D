import sys
import matplotlib.pyplot as plt

def main():
    print("Paste the Pybricks console output below.")
    print("When finished, type 'EOF' on a new line and press Enter.")
    
    t_vals, rx_vals, ry_vals, ax_vals, ay_vals = [], [], [], [], []
    
    while True:
        try:
            line = input().strip()
            if line == 'EOF':
                break
            if line.startswith("LOG,"):
                parts = line.split(',')
                if len(parts) == 6:
                    # Format: LOG, t, x, y, xr, yr
                    t = float(parts[1])
                    ax = float(parts[2])
                    ay = float(parts[3])
                    rx = float(parts[4])
                    ry = float(parts[5])
                    
                    t_vals.append(t)
                    ax_vals.append(ax)
                    ay_vals.append(ay)
                    rx_vals.append(rx)
                    ry_vals.append(ry)
        except EOFError:
            break

    if not t_vals:
        print("No LOG lines found. Use ramsete.py with debug=True for tracking logs.")
        return

    # Plot X/Y Tracking
    plt.figure(figsize=(10, 8))
    plt.plot(rx_vals, ry_vals, label='Reference Trajectory (Optimizer)', linestyle='--', color='gray')
    plt.plot(ax_vals, ay_vals, label='Actual Odometry (Robot)', color='blue')
    
    plt.title('Ramsete Tracking Accuracy')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
