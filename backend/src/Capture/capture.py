import subprocess
import logging

def capture_packets_tshark(capture_duration):
    try:
        command = f'"C:\\Program Files\\Wireshark\\tshark.exe" -i Wi-Fi -a duration:{capture_duration} -w "C:\\Users\\Vishruth V Srivatsa\\OneDrive\\Desktop\\IDS\\backend\\src\\models\\capture.pcap"'
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Capture error: {e}")

def capture_packets_tshark_wrapper(capture_duration):
    try:
        capture_packets_tshark(capture_duration)
    except Exception as e:
        logging.error(f"Capture error: {str(e)}")