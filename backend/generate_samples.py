import requests
import random
import os

with open(r"e:\RGEM\test_samples.txt", "w", encoding="utf-8") as f:
    f.write("Landslide Prediction Model - 100 Sample Tests\n")
    f.write("="*90 + "\n\n")
    
    for i in range(1, 101):
        # Generate random inputs
        rain = round(random.uniform(0, 400), 1)
        # Weight slope a bit more realistically (0-60)
        slope = round(random.uniform(0, 60), 1)
        elev = round(random.uniform(10, 4000), 1)
        ndvi = round(random.uniform(0.0, 0.8), 2)
        clay = round(random.uniform(5, 60), 1)
        
        # 50% chance of earthquake triggering logic
        has_eq = random.choice([True, False])
        if has_eq:
            mag = round(random.uniform(3.5, 8.0), 1)
            depth = round(random.uniform(5, 50), 1)
            pga = round(random.uniform(0.05, 1.2), 3)
            mmi = random.randint(3, 10)
        else:
            mag = 0.0
            depth = 10.0
            pga = 0.0
            mmi = 1
        
        payload = {
            "rainfall": rain,
            "slope": slope,
            "elevation": elev,
            "ndvi": ndvi,
            "soil_moisture": clay,
            "magnitude": mag,
            "depth": depth,
            "pga": pga,
            "mmi": float(mmi)
        }
        
        try:
            r = requests.post("http://127.0.0.1:8001/predict", json=payload)
            res = r.json()
            prob = res.get("probability", 0)
            
            if prob >= 35.0:
                risk = "HIGH"
            elif prob >= 20.0:
                risk = "MODERATE"
            else:
                risk = "LOW"
                
            f.write(f"Test Case #{i}\n")
            f.write(f"Inputs: Rain={rain}mm, Slope={slope}°, Elev={elev}m, NDVI={ndvi}, Clay={clay}%, Mag={mag}, Depth={depth}km, PGA={pga}g, MMI={mmi}\n")
            f.write(f"Result: {prob}% --> {risk} RISK\n")
            f.write("-" * 70 + "\n")
            
        except Exception as e:
            f.write(f"Error on Test #{i}: {e}\n")
