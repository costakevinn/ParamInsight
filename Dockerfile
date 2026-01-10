# Base image Python 3.11 slim
FROM python:3.11-slim

# Configurar diretório de trabalho dentro do container
WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo o código
COPY . .

# Comando padrão ao iniciar o container
CMD ["python", "main.py"]
