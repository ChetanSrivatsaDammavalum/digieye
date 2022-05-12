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

# Pin Definitons:
#led_pin = 12  # Board pin 12
but_pin_blue = 21  # Board pin 21
but_pin_green = 18  # Board pin 18
but_pin_yellow = 15  # Board pin 15

def main():
    # Pin Setup:
    GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme
    #GPIO.setup(led_pin, GPIO.OUT)  # LED pin set as output
    GPIO.setup(but_pin_blue, GPIO.IN)  # button pin set as input
    GPIO.setup(but_pin_green, GPIO.IN)  # button pin set as input
    GPIO.setup(but_pin_yellow, GPIO.IN)  # button pin set as input

    # Initial state for LEDs:
    #GPIO.output(led_pin, GPIO.LOW)
    prev_value = None

 

    print("Starting demo now! Press CTRL+C to exit")
    print("Waiting for button event")
    try:
        while True:

            button_blue = GPIO.input(but_pin_blue)
            button_green = GPIO.input(but_pin_green) 
            button_yellow = GPIO.input(but_pin_yellow) 
            
            if button_blue == 0 :
                print("Blue Button Pressed!") 
            if button_green == 0 :
                print("Green Button Pressed!") 
            if button_yellow == 0 :
                print("Yellow Button Pressed!") 
                print("Requested for shutdown")
                time.sleep(0.2)

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
