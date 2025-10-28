"""
Prerequisites:

    pip install cobs
    pip install pymavlink
    pip install schedule

Usage:

    - Define your messages in `messages.py`
    - Define your state_machines in `state_machines.py`
    - Handle your messages in handle_messages()

Features:

    Allows for both /dev/tty* serial and also TCP port (for simulations)

    The TCP will be useful to trigger simulation actions based on a "simulated"
    companion-mcu.

Testing:

    To evaluate this software before comitting it, both
    the tty Serial and TCP port options were tested.

    For tty Serial, we upload this .ino sketch to the drybox companion-mcu and we connect
    a USB to the Serial bridge we created between the Lua script driver on the autopilot
    and this python script on /dev/ttyUSB0

        void setup() {
            // Start both Serial and Serial1 at the same baud rate
            Serial.begin(57600);   // USB Serial (connected to your computer)
            Serial1.begin(57600);  // Hardware Serial (connected to the peripheral)
        }

        void loop() {
            // If data is available on Serial (USB), send it to Serial1
            if (Serial.available()) {
                while (Serial.available()) {
                char c = Serial.read();  // Read a character from Serial
                Serial1.write(c);        // Write that character to Serial1
                }
            }

            // If data is available on Serial1 (UART), send it to Serial (USB)
            if (Serial1.available()) {
                while (Serial1.available()) {
                char c = Serial1.read();  // Read a character from Serial1
                Serial.write(c);          // Write that character to Serial (USB)
                }
            }
        }
"""


import serial
import socket
import schedule
import time
from cobs import cobs
import struct
import logging
from pymavlink.dialects.v20.common import *
from utilities import *
from messages import *
from state_machines import *
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DRIVER_VERSION = "1.0"

PORT_TYPE               = 'TCP' # 'tty', 'TCP'

if PORT_TYPE == 'tty':
    SERIAL_PORT             = '/dev/ttyACM0'  # typically, /dev/ttyUSB0 or /dev/ttyACM0
    SERIAL_BAUDRATE         = 57600

elif PORT_TYPE == 'TCP':
    TCP_PORT                = 5783  # SITL Serial2
    TCP_HOST                = '127.0.0.1'

else:
    raise SystemExit("Invalid PORT_TYPE. Exiting...")

MAIN_LOOP_PERIOD_S      = 0.05

HEARTBEAT_PERIOD_S      = 1
HEARTBEAT_TIMEOUT_S     = 3
REPORTING_PERIOD_S      = 2
STATE_UPDATE_PERIOD_S   = 0.1

TARGET_SYSTEM         = 2                               # System ID for the Arduino
TARGET_COMPONENT      = MAV_COMP_ID_PERIPHERAL          # Component ID 158
MAV_TYPE              = MAV_TYPE_ONBOARD_CONTROLLER     # MAV_TYPE (18): Onboard companion
MAV_AUTOPILOT         = MAV_AUTOPILOT_INVALID           # MAV_AUTOPILOT (8): No valid autopilot

class Device_State:
    def __init__(self):
        self.state = MAV_STATE_UNINIT
        self.connected = False

device = Device_State()

last_rcvd_heartbeat_s = time.time()

# CUSTOM MAVlink Defines
NV_CMD_REQUEST_NV_STATE         = 1
NV_CMD_DO_LOCK                  = 2
NV_CMD_DO_CUSTOM_ARM            = 3
NV_CMD_DO_MOTOR_MODE            = 4
NV_CMD_DO_TRANSITION            = 5
NV_CMD_DO_GUIDED_SURFACE        = 6
NV_CMD_DO_GUIDED_DIVE           = 7
NV_CMD_DO_UW_ARM_TO_AUTO        = 8

def handle_messages(conn, struct_type, message_data):
    global last_rcvd_heartbeat_s

    if struct_type == MAVLINK_MSG_ID_HEARTBEAT:

        msg = Heartbeat.unpack(message_data)
        logging.debug(f"Received Heartbeat: {msg}")

        # update device state
        if not device.connected:
            logging.info("HEARBEAT timeout cleared")
            device.connected = True
        
        last_rcvd_heartbeat_s = time.time()
    
    elif struct_type == MAVLINK_MSG_ID_COMMAND_LONG:

        msg = CommandLong.unpack(message_data)
        logging.info(f"Received CommandLong: {msg}")

        # this is an example, only use if this capability is supported
        if msg.command == MAV_CMD_DO_PARACHUTE and msg.param1 == PARACHUTE_RELEASE:
            # do some action

            # send ack
            msg_out = CommandAck()
            msg_out.command = MAV_CMD_DO_PARACHUTE
            msg_out.result = MAV_RESULT_ACCEPTED
            send_message(ser, msg_out, MAVLINK_MSG_ID_COMMAND_ACK)
    
    elif struct_type == MAVLINK_MSG_ID_COMMAND_ACK:

        msg = CommandAck.unpack(message_data)
        logging.info(f"Received CommandAck: Command={msg.command}, "
                    f"Result={msg.result}")
        
    else:
        logging.warning(f"Unknown struct type: {struct_type}")

def send_heartbeat(conn):
    heartbeat = Heartbeat()
    send_message(conn, heartbeat, MAVLINK_MSG_ID_HEARTBEAT)

def report():
    if time.time() - last_rcvd_heartbeat_s > HEARTBEAT_TIMEOUT_S:
        logging.warning(f"No HEARTBEAT received, connection lost. ({time.time() - last_rcvd_heartbeat_s:.3f} s > {HEARTBEAT_TIMEOUT_S} s)")
        device.connected = False

    pass

def update_state():
    global last_rcvd_heartbeat_s

    if device.connected and time.time() - last_rcvd_heartbeat_s > HEARTBEAT_TIMEOUT_S:
        logging.warning(f"HEARTBEAT timeout")
        device.connected = False

schedule.every(1).seconds.do(lambda: send_heartbeat(ser if PORT_TYPE == 'tty' else conn))
schedule.every(2).seconds.do(report)
schedule.every(0.1).seconds.do(update_state)

# Abstraction for reading data (works for both serial and TCP)
def read_data(conn):
    if PORT_TYPE == 'tty':
        return conn.read(1024)
    else:
        try:
            data = conn.recv(1024)
            if data:
                logging.debug(f"Received data: {data}")
                return data
        except socket.error as e:
            logging.error(f"Socket error: {e}")
        return b''

# Abstraction for sending data
def send_data(conn, data):
    if PORT_TYPE == 'tty':
        conn.write(data)
    else:
        conn.sendall(data)

# Main function
def main():
    print(f"companion-mcu-driver: Running v{DRIVER_VERSION}")

    global ser, conn

    if PORT_TYPE == 'tty':
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=0)
        conn = ser  # Alias to keep common logic

    elif PORT_TYPE == 'TCP':
        tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        logging.info(f"Connecting to {TCP_HOST}:{TCP_PORT}")
        tcp_sock.connect((TCP_HOST, TCP_PORT))
        conn = tcp_sock

    buffer = bytearray()
    
    # Give some time for the serial connection to stabilize
    time.sleep(2)

    try:
        StateMachine.init_all()
        device.state = MAV_STATE_ACTIVE
        last_t = time.time()
        while True:
            now = time.monotonic()
            data = read_data(conn)
            if data:
                buffer.extend(data)
                handle_packet(conn, buffer, handle_messages)
                              
            StateMachine.run_all()
            schedule.run_pending()

            # Calculate sleep time to avoid CPU high usage
            next_wakeup = now + MAIN_LOOP_PERIOD_S
            sleep_time = max(0, next_wakeup - time.monotonic())
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logging.info("Terminating...")

    finally:
        if PORT_TYPE == 'tty':
            ser.close()

        elif PORT_TYPE == 'TCP':
            conn.close()


if __name__ == "__main__":
    main()