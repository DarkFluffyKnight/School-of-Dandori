FROM python:3.11-slim
WORKDIR /app
# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy application files
COPY . .
# Expose port (Cloud Run will set this via $PORT)
EXPOSE 8080
# Create .streamlit directory and config
RUN mkdir -p ~/.streamlit
RUN echo "\
[server]\n\
headless = true\n\
port = 8080\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
" > ~/.streamlit/config.toml
# Run streamlit
CMD streamlit run app.py --server.port=8080 --server.address=0.0.0.0