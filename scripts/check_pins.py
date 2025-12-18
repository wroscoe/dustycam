#!/usr/bin/env python3
import time
import sys
from gpiozero import Button

# Configuration
# Standard GPIO pins on Raspberry Pi (BCM numbering)
GPIO_PINS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]

def monitor():
    print("Initializing pins... (Ctrl+C to stop)")
    buttons = {}
    
    # Initialize all pins
    for pin in GPIO_PINS:
        try:
            # pull_up=True means Open=High (1), Grounded=Low (0)
            buttons[pin] = Button(pin, pull_up=True)
        except Exception as e:
            print(f"Warning: Could not initialize GPIO {pin}: {e}")
            buttons[pin] = None

    print("\033[2J") # Clear screen once
    
    try:
        while True:
            # Move cursor to top-left
            sys.stdout.write("\033[H")
            
            print("=== GPIO Pin Monitor ===")
            print("State: ON = OPEN (Pull-up High), OFF = GROUNDED (Low)")
            print("-" * 40)
            
            # Print in two columns
            col_len = (len(GPIO_PINS) + 1) // 2
            
            for i in range(col_len):
                # Left column
                pin1 = GPIO_PINS[i]
                b1 = buttons.get(pin1)
                s1 = "ERR "
                if b1:
                    # value 1 means Open (High), 0 means Grounded (Low)
                    val1 = b1.value 
                    s1 = "OPEN" if val1 else "GND "
                    # Color: Green for GND (active), Red for OPEN (inactive)
                    color1 = "\033[92m" if not val1 else "\033[91m"
                    reset = "\033[0m"
                    s1 = f"{color1}{s1}{reset}"
                
                line = f"GPIO {pin1:02d}: {s1}    "
                
                # Right column
                if i + col_len < len(GPIO_PINS):
                    pin2 = GPIO_PINS[i + col_len]
                    b2 = buttons.get(pin2)
                    s2 = "ERR "
                    if b2:
                        val2 = b2.value
                        s2 = "OPEN" if val2 else "GND "
                        color2 = "\033[92m" if not val2 else "\033[91m"
                        reset = "\033[0m"
                        s2 = f"{color2}{s2}{reset}"
                    line += f"|    GPIO {pin2:02d}: {s2}"
                
                print(line + "\033[K") # Clear rest of line

            print("-" * 40)
            print("Press Ctrl+C to exit.\033[J") # Clear rest of screen below
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        for b in buttons.values():
            if b:
                b.close()

if __name__ == '__main__':
    monitor()
