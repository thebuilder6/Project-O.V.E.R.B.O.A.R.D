"""
ROBOT SYSTEM IDENTIFICATION (SysId) TOOL
----------------------------------------
This script executes the Automated SysId Test Protocol to generate 
nominal and interval parameters for the JAX/Immrax PC-side optimizer.

Data is printed to the console in CSV format. 
To use: Run this script, follow the on-hub display/console instructions,
copy the console output, and save it as a .csv file on your PC.
"""

from pybricks.hubs import PrimeHub
from pybricks.pupdevices import Motor, ColorSensor, UltrasonicSensor
from pybricks.parameters import Button, Color, Direction, Port, Side, Stop, Axis
from pybricks.robotics import DriveBase
from pybricks.tools import wait, StopWatch

# ------------------------------------------------------------------
# HUB & HARDWARE INITIALIZATION (From your config)
# ------------------------------------------------------------------
hub = PrimeHub(top_side=Axis.Z, front_side=Axis.Y)


# Reassign the emergency stop to the Bluetooth button. 
# This frees up Button.CENTER to be used as our standard "Next/OK" input
# without accidentally killing the script. If you prefer no stop button at all, 
# you can use hub.system.set_stop_button(None).
hub.system.set_stop_button((Button.BLUETOOTH,))

# Drive motors (Left reversed to move forward positively)
motorLeft = Motor(Port.F, Direction.COUNTERCLOCKWISE)
motorRight = Motor(Port.B, Direction.CLOCKWISE)

# Disable DriveBase for SysId so we can control motors directly via Voltage/Duty
# drivebase = DriveBase(motorLeft, motorRight, wheel_diameter=56, axle_track=96.5)

watch = StopWatch()

# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------
def wait_for_start(test_name, instructions):
    """Pauses the program until the center button is pressed."""
    hub.light.on(Color.YELLOW)
    print("\n" + "="*50)
    print(f"READY FOR: {test_name}")
    print(f"SETUP: {instructions}")
    print("ACTION: Press the CENTER button to begin logging.")
    print("        (Press the BLUETOOTH button to emergency stop)")
    print("="*50 + "\n")
    
    # Wait for center button to be pressed
    while Button.CENTER not in hub.buttons.pressed():
        wait(10)
        
    hub.display.text("OK") # Visual feedback that the press registered
    
    # Wait for center button to be released
    while Button.CENTER in hub.buttons.pressed():
        wait(10)
        
    hub.display.off()
    hub.light.on(Color.RED)
    wait(500) # Give user a half-second to let go of the robot

def run_loop_at_target_dt(start_time, target_ms=10):
    """Ensures a stable logging frequency (100Hz default)"""
    elapsed = watch.time() - start_time
    slack = target_ms - elapsed
    if slack > 0:
        wait(slack)

# ------------------------------------------------------------------
# SYSID TEST PROTOCOLS
# ------------------------------------------------------------------

def test_a_backlash():
    wait_for_start("TEST A: Gear Backlash (b)", 
                   "Place robot on HIGH-TRACTION surface (mat).")
    
    print("TEST_A_START")
    print("time_ms,l_angle_deg,r_angle_deg,gyro_z_rad_s")
    
    watch.reset()
    while watch.time() < 5000: # Run for 5 seconds
        loop_start = watch.time()
        
        # Oscillate every 500ms
        cycle = (watch.time() // 500) % 2
        duty = 15 if cycle == 0 else -15 # Tiny duty cycle to cross the deadband
        
        motorLeft.dc(duty)
        motorRight.dc(duty)
        
        # Pybricks gyro returns deg/s, convert to rad/s for JAX/CasADi
        gyro_z = hub.imu.angular_velocity()[2] * (3.14159 / 180.0)
        
        # Print CSV row
        print(f"{watch.time()},{motorLeft.angle()},{motorRight.angle()},{gyro_z:.4f}")
        
        run_loop_at_target_dt(loop_start, 10)
        
    motorLeft.stop()
    motorRight.stop()
    print("TEST_A_END")

def test_b_friction():
    wait_for_start("TEST B: Friction Coefficient (mu)", 
                   "Place robot's FRONT BUMPER flat against a RIGID WALL.")
    
    print("TEST_B_START")
    print("time_ms,duty_pct,l_speed_deg_s,r_speed_deg_s,l_load_mNm,r_load_mNm,accel_x_g,accel_y_g")
    
    watch.reset()
    while watch.time() < 3000: # Ramp up over 3 seconds
        loop_start = watch.time()
        
        # Linearly ramp duty from 0 to 100
        duty = (watch.time() / 3000.0) * 100.0
        motorLeft.dc(duty)
        motorRight.dc(duty)
        
        # Pybricks acceleration is in mm/s^2, convert to standard g's
        accel = hub.imu.acceleration()
        ax, ay = accel[0] / 9810.0, accel[1] / 9810.0
        
        print(f"{watch.time()},{duty:.1f},{motorLeft.speed()},{motorRight.speed()},{motorLeft.load()},{motorRight.load()},{ax:.4f},{ay:.4f}")
        
        run_loop_at_target_dt(loop_start, 10)
        
    motorLeft.stop()
    motorRight.stop()
    print("TEST_B_END")

def test_c_inertia():
    wait_for_start("TEST C: Moment of Inertia (I)", 
                   "Place robot in OPEN SPACE on the mat.")
    
    print("TEST_C_START")
    print("time_ms,duty_pct,gyro_z_rad_s,l_load_mNm,r_load_mNm,l_speed_deg_s,r_speed_deg_s")
    
    watch.reset()
    while watch.time() < 1500: # Spin for 1.5 seconds
        loop_start = watch.time()
        
        # Step response (Left forward, Right backward)
        duty = 100 
        motorLeft.dc(duty)
        motorRight.dc(-duty)
        
        gyro_z = hub.imu.angular_velocity()[2] * (3.14159 / 180.0)
        
        print(f"{watch.time()},{duty},{gyro_z:.4f},{motorLeft.load()},{motorRight.load()},{motorLeft.speed()},{motorRight.speed()}")
        
        run_loop_at_target_dt(loop_start, 10)
        
    motorLeft.stop()
    motorRight.stop()
    print("TEST_C_END")

def test_d_motor_map():
    wait_for_start("TEST D: Motor Performance Mapping", 
                   "Ensure robot has 2+ METERS of clear straight runway.")
    
    print("TEST_D_START")
    print("time_ms,l_speed_deg_s,r_speed_deg_s,l_load_mNm,r_load_mNm,batt_v_mV,batt_i_mA")
    
    watch.reset()
    while watch.time() < 3000: # 2s max speed, 1s hard brake
        loop_start = watch.time()
        t = watch.time()
        
        if t < 2000:
            motorLeft.dc(100)
            motorRight.dc(100)
        else:
            # Force stall to measure maximum reverse-current/torque
            motorLeft.brake()
            motorRight.brake()
            
        print(f"{t},{motorLeft.speed()},{motorRight.speed()},{motorLeft.load()},{motorRight.load()},{hub.battery.voltage()},{hub.battery.current()}")
        
        run_loop_at_target_dt(loop_start, 10)
        
    motorLeft.stop()
    motorRight.stop()
    print("TEST_D_END")

# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------
if __name__ == "__main__":
    hub.light.on(Color.BLUE)
    print("Starting SysId Diagnostics...")
    
    test_a_backlash()
    test_b_friction()
    test_c_inertia()
    test_d_motor_map()
    
    hub.light.on(Color.GREEN)
    print("\nSysId Data Collection Complete!")
    print("Copy the CSV blocks above into your PC SysId solver.")
