#!/bin/bash
set -e

# Clean up any stale PID files from previous runs
# Without this, restarting the container fails
echo "Cleaning up any existing Airflow processes..."
pkill -f "airflow webserver" || true
pkill -f "airflow scheduler" || true
rm -f /opt/airflow/airflow-webserver.pid
rm -f /opt/airflow/airflow-scheduler.pid

sleep 2

# Init the Airflow metadata database (creates tables)
echo "Initializing Airflow database..."
airflow db init

# Create the admin user — credentials: admin/admin
echo "Creating admin user..."
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin || echo "Admin user already exists"

# Start both webserver and scheduler in the same container
# For production you'd run these as separate services
echo "Starting Airflow webserver and scheduler..."
airflow webserver --port 8080 --daemon &
airflow scheduler