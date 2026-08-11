# CAR
Raspberry Pi Autonomous Driving & Delivery Vehicle

This repository contains a reconstructed version of a 2022–2023 university capstone project.

The original source code was lost, so this repository was rebuilt from the remembered system architecture, project workflow, hardware configuration, and driving logic. It should therefore be treated as a technical reconstruction rather than the exact original source.

Project concept

The vehicle evolved from an Arduino-based indoor delivery prototype into a Raspberry Pi-based outdoor driving platform.

The reconstructed system combines:

Raspberry Pi camera input

CNN-based road / obstacle-level image classification

Two infrared line-tracking sensors

Ultrasonic obstacle-distance sensing

Left/right DC motor PWM control

Manual driving-image labeling and data collection

Periodic CNN retraining

Control architecture

Camera
  |
  v
CNN road / obstacle classification
  |
  v
Camera speed limit ----------------------+
                                         |
Ultrasonic sensor                        |
  |                                      |
  v                                      |
Distance-based speed limit --------------+
                                         |
                                         v
                                 Safe base speed
                                         |
                                         v
                              IR Left / IR Right
                                         |
                                         v
                              Direction correction
                                  /           \
                                 v             v
                           Left PWM       Right PWM
                                 \             /
                                  v           v
                                      Vehicle

Remembered speed-control concept

Camera / CNN

Detected condition

Speed

Normal road

50

Obstacle level 1

40

Obstacle level 2

30

Obstacle level 3

20

Obstacle level 4

10

Obstacle level 5

Stop

Ultrasonic sensor

Distance

Speed

> 50 cm

50

40–50 cm

40

30–40 cm

30

20–30 cm

20

10–20 cm

10

<= 10 cm

Stop

The reconstructed control logic uses the more conservative speed limit from the camera/CNN and ultrasonic sensor.

IR line tracking

Two IR sensors were mounted on the left and right sides of the vehicle to distinguish the white lane boundaries from the black road surface.

The sensor modules' onboard variable resistors were manually adjusted during repeated road testing to improve black/white detection sensitivity.

The current reconstruction assumes:

Left BLACK + Right BLACK -> Forward
Left WHITE + Right BLACK -> Right correction
Left BLACK + Right WHITE -> Left correction
Left WHITE + Right WHITE -> Stop / line lost

Sensor polarity and correction direction may need to be reversed depending on the actual sensor module and mechanical layout.

Data collection and CNN retraining

While the program is running, keys 0 to 5 can be used to manually label the current driving condition.

0 -> normal
1 -> obstacle_1
2 -> obstacle_2
3 -> obstacle_3
4 -> obstacle_4
5 -> obstacle_5
x -> stop labeling
t -> retrain CNN
q -> quit

While a label is active, camera frames are periodically saved into the corresponding dataset folder.

After a specified number of newly labeled images is collected, the vehicle stops and the CNN is retrained.

Dataset structure

dataset/
├── normal/
├── obstacle_1/
├── obstacle_2/
├── obstacle_3/
├── obstacle_4/
└── obstacle_5/

Hardware represented in the reconstruction

Raspberry Pi

Camera

Two IR line-tracking sensors

Ultrasonic distance sensor

DC motors

Motor driver

PWM motor control

The original vehicle chassis and sensor layout were designed by referencing real black road surfaces and white lane markings, modeled with CAD / Autodesk Inventor, and fabricated using 3D printing.

Important

The GPIO assignments, CNN architecture, retraining interval, sensor polarity, and several implementation details are reconstructed assumptions.

Do not connect the vehicle and run this code without first checking the actual wiring and testing the motor outputs with the wheels lifted off the ground.

Main file

autonomous_vehicle.py
