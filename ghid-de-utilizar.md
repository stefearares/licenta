# Ghid de utilizare – Procesare Imagini Satelitare & Analiză

### Obs. Un fișier cu datele utilizate în partea scrisă este disponibil în folder-ul "rezultate"

##  Instalare dependințe

~~~bash
pip install -r requirements.txt
~~~

##  Pornirea API-ului FLask

~~~bash
python web_server.py
~~~

## 🌐 Endpoint-uri disponibile

- `GET  /` – Arată toate endpoint-urile existente  
- `GET  /health` – Health check  
- `GET  /process-images` – Procesare imagini satelitare (simulare)  
- `GET  /process-images-real` – Process imagini satelitare (real)  
- `GET  /process-normal-images-hardcoded` – Proceseaza setul de test  
- `GET  /process-safe-folder` – Proceseaza un .SAFE folder
- `GET  /plot-bar-evolution` – Genereaza bar-evolution plot  
- `GET  /compare-csv` – Analiză ARIMA  
- `POST /set-safe-folder` – Setează path folder .SAFE  
- `POST /set-csv-file` – Setează path fișier CSV  

### 🔗 Exemple URL-uri

~~~text
http://localhost:5000/process-normal-images-hardcoded
http://localhost:5000/process-safe-folder-with-export?folder_path=/path/to/safe/folder&threshold=75.0&export_folder=/path/to/export
http://localhost:5000/process-safe-folder?folder_path=/path/to/safe/folder&threshold=75.0
http://localhost:5000/process-images-real?image_folder=/path/to/images&threshold=85.0
http://localhost:5000/plot-bar-evolution?csv_path=/path/to/data.csv&plot_type=sarima
http://localhost:5000/compare-csv?csv_path=/path/to/data.csv&forecast_steps=10
~~~

##  Rulare teste – ARIMA / SARIMA / Auto-ARIMA

### Se deschide un nou terminal pentru rularea testelor unde se pot folosi comenzile
~~~bash
python test_script.py path/to/your.csv --window 3 --train-until 2020 --test-until 2023 --test-stationarity
~~~

###  Argumente disponibile

- `csv` – path la CSV **(obligatoriu)**  
- `--window` – sliding-window marime (default: 3)  
- `--train-until` – ultimul an inclus la training (hold-out)  
- `--test-until` – ultimul an pentru care se va face predicția (hold-out)  
- `--test-stationarity` – teste de staționaritate pe date

### Obs. În cazul problemelor tehnice este posibilă rularea fiecărei părți a codului prin eliminarea comentariilor din funcția dorită în fișierul <i>main.py</i>