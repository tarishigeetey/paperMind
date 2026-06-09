FROM python:3.12-slim

# Airflow config
ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW_VERSION=2.10.3
ENV PYTHON_VERSION=3.12
ENV CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

# System dependencies
# poppler-utils + tesseract for PDF processing in Week 2
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libpq-dev \
        poppler-utils \
        tesseract-ocr \
        && rm -rf /var/lib/apt/lists/*

# Create airflow user with UID 50000
# Standard Airflow convention — same UID across all environments
RUN groupadd -r -g 50000 airflow && \
    useradd -r -u 50000 -g airflow -d ${AIRFLOW_HOME} -s /bin/bash airflow

# Create directories with proper ownership
RUN mkdir -p ${AIRFLOW_HOME}/dags \
             ${AIRFLOW_HOME}/logs \
             ${AIRFLOW_HOME}/plugins && \
    chown -R 50000:50000 ${AIRFLOW_HOME} && \
    chmod -R 755 ${AIRFLOW_HOME}

# Install Airflow with PostgreSQL support
# --constraint pins all transitive dependencies — prevents version conflicts
RUN pip install --no-cache-dir \
    "apache-airflow[postgres]==${AIRFLOW_VERSION}" \
    --constraint "${CONSTRAINT_URL}" \
    psycopg2-binary

# Install project-specific dependencies
COPY requirements-airflow.txt /tmp/requirements-airflow.txt
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt

# Copy entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER airflow
WORKDIR ${AIRFLOW_HOME}

EXPOSE 8080

CMD ["/entrypoint.sh"]