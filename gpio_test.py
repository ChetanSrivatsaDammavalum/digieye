#!/usr/bin/env python

 
#taken from Jetson.GPIO github examples

# EXAMPLE SETUP
# Connect a button to pin 18 and GND, a pull-up resistor connecting the button
# to 3V3 and an LED connected to pin 12. The application performs the same
# function as the button_led.py but performs a blocking wait for the button
# press event instead of continuously checking the value of the pin in order to
# reduce CPU usage.

import RPi.GPIO as GPIO
import time
import os

# Pin Definitons: 13, 15,19,21,23 - new pin list
#led_pin = 12  # Board pin 12
but_pin_13 = 13  # Board pin 13
but_pin_15 = 15  # Board pin 15
but_pin_19 = 19  # Board pin 1119
but_pin_19 = 21
but_pin_19 = 23


def main():
    # Pin Setup:
    GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme
    #GPIO.setup(led_pin, GPIO.OUT)  # LED pin set as output
    GPIO.setup(but_pin_13, GPIO.IN)  # button pin set as input
    GPIO.setup(but_pin_15, GPIO.IN)  # button pin set as input
    GPIO.setup(but_pin_19, GPIO.IN)  # button pin set as input
    GPIO.setup(but_pin_21, GPIO.IN)  # button pin set as input
    GPIO.setup(but_pin_23, GPIO.IN)  # button pin set as input

    # Initial state for LEDs:
    #GPIO.output(led_pin, GPIO.LOW)
    prev_value = None

 

    print("Starting demo now! Press CTRL+C to exit")
    print("Waiting for button event")
    try:
        while True:

            button_plus = GPIO.input(but_pin_13)
            button_minus = GPIO.input(but_pin_15) 
            button_mode = GPIO.input(but_pin_19) 
            button_submode = GPIO.input(but_pin_21) 
            button_shutdown = GPIO.input(but_pin_23) 
            
            if
             button_plus == 0 :
                print(" (+) Pressed!") 
            if button_minus == 0 :
                print(" (-) Pressed!") 
            if button_mode == 0 :
                print(" (M) Pressed!") 
            if button_submode == 0 :
                print(" (M) Pressed!") 
            if button_shutdown == 1 :
                print(" (P) Pressed!") 
                print("Shutdown initiated")
                time.sleep(2)
                print("SYSTEM SHUTDOWN")
                break
            time.sleep(0.2)
    finally:
        GPIO.cleanup()  # cleanup all GPIOs
        

if __name__ == '__main__':
    main()
    #os.system('shutdown /s /t 1') #error:Failed to parse time specification: /s
    print("Shutting down")
    time.sleep(1)
    os.system('shutdown -h now')
