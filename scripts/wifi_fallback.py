#!/usr/bin/env python3
import time
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_internet(timeout=10):
    """Checks for internet connection by pinging 8.8.8.8."""
    logging.info(f"Checking for internet connection for {timeout} seconds...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # Ping Google's DNS server once
            subprocess.check_call(['ping', '-c', '1', '-W', '1', '8.8.8.8'], 
                                  stdout=subprocess.DEVNULL, 
                                  stderr=subprocess.DEVNULL)
            logging.info("Internet connection detected.")
            return True
        except subprocess.CalledProcessError:
            time.sleep(1)
            continue
    logging.info("No internet connection detected.")
    return False

def create_hotspot():
    """Creates a WiFi hotspot using nmcli if it doesn't exist."""
    con_name = "DustyCam"
    ssid = "DustyCam"
    password = "dustycam" 
    
    # Check if connection exists
    connection_exists = False
    try:
        subprocess.check_call(['nmcli', 'con', 'show', con_name],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
        connection_exists = True
    except subprocess.CalledProcessError:
        pass

    if not connection_exists:
        logging.info(f"Creating hotspot '{con_name}'...")
        try:
            # 1. Add the connection
            # autoconnect=no is crucial so it doesn't fight for control on boot
            subprocess.check_call([
                'nmcli', 'con', 'add', 
                'type', 'wifi', 
                'ifname', 'wlan0', 
                'con-name', con_name, 
                'autoconnect', 'no', 
                'ssid', ssid
            ])
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to create hotspot connection: {e}")
            return

    # Update settings (works for both new and existing connections)
    try:
        cmd = [
            'nmcli', 'con', 'modify', con_name, 
            '802-11-wireless.mode', 'ap', 
            '802-11-wireless.band', 'bg', 
            'ipv4.method', 'shared',
            'ipv4.addresses', '10.42.0.1/24',  # Explicitly set IP
            'wifi-sec.key-mgmt', 'wpa-psk', 
            'wifi-sec.psk', password
        ]
        
        subprocess.check_call(cmd)
        logging.info(f"Hotspot '{con_name}' configured successfully (IP: 10.42.0.1).")
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to configure hotspot: {e}")

def activate_hotspot():
    """Activates the hotspot connection."""
    con_name = "DustyCam"
    try:
        # Disconnect wlan0 from whatever it's failing to connect to
        subprocess.call(['nmcli', 'device', 'disconnect', 'wlan0'], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        logging.info(f"Activating hotspot '{con_name}'...")
        subprocess.check_call(['nmcli', 'con', 'up', con_name])
        logging.info(f"Hotspot '{con_name}' active.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to activate hotspot: {e}")

def ensure_hotspot_off():
    """Forces the hotspot down to free up the wifi card for client mode."""
    con_name = "DustyCam"
    try:
        # We try to down it regardless of state to be sure wlan0 is free
        # Suppress output because it might fail if already down (which is fine)
        subprocess.call(['nmcli', 'con', 'down', con_name], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        logging.info(f"Ensured Hotspot '{con_name}' is OFF.")
    except Exception as e:
        logging.error(f"Error turning off hotspot: {e}")

def main():
    # 1. Setup profiles
    create_hotspot()

    # 2. Force Hotspot OFF. 
    # This releases wlan0 so NetworkManager can try to connect to your home WiFi.
    ensure_hotspot_off()

    # 3. Wait for NetworkManager to Auto-Connect
    # If the hotspot was just on, it takes 10-20s for the Pi to scan 
    # and re-connect to a known home network.
    logging.info("Waiting 20s for NetworkManager to attempt client connection...")
    time.sleep(20)

    # 4. Check for internet
    if check_internet(timeout=10):
        logging.info("Internet is working. Staying in Client Mode.")
    else:
        # 5. If no internet, switch back to Hotspot
        logging.info("No internet found. Switching to Hotspot Mode.")
    
        activate_hotspot()

if __name__ == "__main__":
    main()