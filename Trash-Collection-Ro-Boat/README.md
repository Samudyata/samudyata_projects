# ♻️ Trash Collection Ro-Boat

An autonomous, low-cost aquatic robot designed to detect, collect, and manage surface trash from water bodies using image processing, smart sensors, and IoT-enabled navigation. Built to address the growing problem of water pollution, especially from plastics and non-biodegradable waste.

---

## 📌 Project Overview

🌊 **Problem**  
Water pollution is escalating globally. Much of this waste floats and accumulates near shorelines, posing environmental and health threats.

🚤 **Solution**  
We developed a smart, semi-autonomous robot boat that:
- Detects and collects floating garbage using Raspberry Pi and image processing.
- Differentiates between living and non-living objects using PIR sensors.
- Navigates using BLDC and servo motors.
- Pushes large trash items to the nearest bank and offloads collected trash once a threshold is reached.

---

## 🧠 Key Features

- 🔍 **Real-time Trash Detection** using OpenCV and Pi Camera
- 🧭 **Navigation System** with Servo and BLDC motors
- 📦 **Smart Collection Mechanism** using Stepper Motors and Load Sensors
- 🧠 **Obstacle & Organism Recognition** using PIR sensors
- 🏖️ **Auto-Docking** at the nearest bank once the bin is full
- 🪙 **Low-Cost Design** – Only ₹9,490 compared to commercial setups worth ₹41,000

---

## 🛠️ Tech Stack

### Hardware:
- Raspberry Pi 3B
- Raspberry Pi Camera
- BLDC & Servo Motors
- PIR Sensor
- Weight Sensor (Load Cell)
- LiPo Battery
- Pool Noodles for Floatation

### Software:
- Python
- Raspberry Pi OS IDE
- OpenCV (Image Processing)
- TensorFlow (Object Classification)
- PWM Control via `machine` module

---

## 📈 Architecture

### 🔧 Hardware Architecture

- BLDC motors for forward/backward movement
- Servo motors for navigation
- Stepper motors for trash collection
- Load sensor to track bin capacity
- PIR sensor to detect living organisms

### 💻 Software Architecture

- Object detection and classification using Python + OpenCV
- Motor control and hardware interfacing using Raspberry Pi GPIO
- PWM-based speed and direction management

---


## ✅ Achievements

- Accurate trash detection in water
- Identified and avoided living organisms
- Smart routing and movement
- Cost-effective and modular design

---

## 🔮 Future Work

- Upgrade detection using Deep Learning for object classification (e.g. YOLOv5)
- Solar-based battery recharge system for energy independence
- Add multi-trash classification (plastics, metals, organic)
- Integrate mobile/web dashboard for remote monitoring
