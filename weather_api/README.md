# Weather API

This folder contains:
- `train_and_save_model.py`: Trains a regression model to predict temperature from humidity and pressure, and saves it as `model.pkl`.
- `app.py`: Flask API that loads `model.pkl` and exposes a `/predict` endpoint.
- `requirements.txt`: Required Python packages.

## Setup

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Train and save the model**

```bash
python train_and_save_model.py
```

This will create `model.pkl` in the same folder.

3. **Run the Flask API**

```bash
python app.py
```

The API will be available at `http://127.0.0.1:5000/predict`.

## API Usage

**POST** `/predict`

- **Input JSON:**
  ```json
  {
    "humidity": 55,
    "pressure": 1013
  }
  ```
- **Response JSON:**
  ```json
  {
    "predicted_temperature": 23.45
  }
  ``` 

Let's troubleshoot the "site can't be reached" error for your Flask app. Here are the most common causes and solutions:

---

### 1. **Is the Flask server running?**
- In your terminal, after running `python app.py`, you should see:
  ```
  * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
  ```
- If you see an error, please share it here.

---

### 2. **Are you in the correct directory?**
- Make sure you are in the `weather_api` directory when running `python app.py`.

---

### 3. **Is the model file present?**
- Ensure `model.pkl` exists in the `weather_api` folder. If not, run:
  ```bash
  python train_and_save_model.py
  ```

---

### 4. **Are you opening the correct URL?**
- Open your browser and go to: [http://127.0.0.1:5000/](http://127.0.0.1:5000/) or [http://localhost:5000/](http://localhost:5000/)

---

### 5. **Is another process using port 5000?**
- If port 5000 is in use, Flask may not start. Try changing the port in `app.py`:
  ```python
  if __name__ == '__main__':
      app.run(debug=True, port=5001)
  ```
  Then visit [http://127.0.0.1:5001/](http://127.0.0.1:5001/)

---

### 6. **Firewall or Antivirus**
- Sometimes, firewall or antivirus software blocks local servers. Try disabling them temporarily.

---

### 7. **Check for Errors in Terminal**
- If Flask crashes or shows an error, copy the error message here for help.

---

#### **If you want, I can run the commands to check for you. Would you like me to do that?**
Or, please let me know what you see in your terminal after running `python app.py`. If there is an error, paste it here for a quick fix! 