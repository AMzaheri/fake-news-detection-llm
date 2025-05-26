FROM python:3.10-slim

# Set working directory to something writable
WORKDIR /app

# Set environment variables so Streamlit writes configs to /app/.streamlit
ENV HOME=/app

COPY . /app/

RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit's default port
EXPOSE 7860

# Run Streamlit app
CMD ["streamlit", "run", "app_streamlit/app.py", "--server.port=7860", "--server.enableCORS=false"]

