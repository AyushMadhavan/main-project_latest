# Deploying Police HQ to Render

The **Police HQ Server** is designed to run in the cloud, aggregating data from all your local surveillance cameras.

## 1. Preparation
We have already created the necessary files:
- `Procfile`: Tells Render how to start the app.
- `hq_requirements.txt`: Lightweight dependencies for the server.

## 2. Push to GitHub
If you haven't already, push this code to a GitHub repository.

## 3. Deploy on Render
1.  Create a free account at [render.com](https://render.com).
2.  Click **"New"** -> **"Web Service"**.
3.  Connect your GitHub repository.
4.  Configure the service:
    - **Name**: `eagle-eye-hq` (or similar)
    - **Runtime**: `Python 3`
    - **Build Command**: `pip install -r hq_requirements.txt`
    - **Start Command**: `gunicorn tools.hq_server:app --bind 0.0.0.0:$PORT`
5.  Click **"Create Web Service"**.

## 4. Connect Your Config
Once deployed, Render will give you a URL (e.g., `https://eagle-eye-hq.onrender.com`).

1.  Open your local `settings.yaml`.
2.  Update the `central_server` field:
    ```yaml
    system:
      central_server: "https://eagle-eye-hq.onrender.com"
    ```
3.  Restart `main.py`.

Now, your local detections will be sent securely to the cloud!
