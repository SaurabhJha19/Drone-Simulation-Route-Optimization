# Drone Delivery Route Optimization Simulator 🚁

## 📌 Overview
This project is a graphical simulation of a **drone-based delivery system** that optimizes delivery routes using a **Genetic Algorithm (GA)**. The goal is to minimize total travel distance while dynamically adapting to environmental conditions such as weather and battery constraints.

The application was developed as part of **iHack**, focusing on algorithmic optimization, simulation, and real-time visualization.

---

## ⚙️ Key Features
- 📍 Interactive placement of depot and delivery points  
- 🧬 Genetic Algorithm for route optimization  
- 🌦️ Dynamic weather conditions affecting speed and battery usage  
- 🔋 Battery-aware drone simulation with automatic re-routing  
- 📊 Real-time dashboard displaying distance, time, battery, and generations  
- 🎨 Visual route rendering using Tkinter GUI  

---

## 🛠️ Tech Stack
- **Language:** Python  
- **GUI:** Tkinter  
- **Algorithm:** Genetic Algorithm  
- **Concepts Used:** Optimization, Simulation, OOP, Event-driven programming  

---

## 🧠 How It Works
1. User places a **depot** and multiple **delivery points** on the canvas.
2. A Genetic Algorithm generates and evolves possible routes.
3. The **optimal route** is selected based on minimum distance.
4. The drone follows the route while:
   - Consuming battery
   - Reacting to weather changes
   - Re-routing if battery is low or conditions worsen

---

## ▶️ How to Run
```bash
python drone.py
