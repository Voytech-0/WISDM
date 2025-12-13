#!/bin/bash
set -e

# Allow overriding the Aim probe URL (use /status endpoint by default)
AIM_SERVER_URL=${AIM_SERVER_URL:-http://aim-server:43800/status}
AIM_SERVER_FALLBACK=${AIM_SERVER_FALLBACK:-http://aim-server:43800/}
# Allow overriding the Aim repo URL used by ZenML (use service name by default)
AIM_REPO_URL=${AIM_REPO_URL:-aim://aim-server:43800}

# Wait for Aim server to be reachable before registering the tracker
MAX_TRIES=30
SLEEP_SECONDS=2
for i in $(seq 1 $MAX_TRIES); do
    echo "Probing Aim server at $AIM_SERVER_URL (attempt $i/$MAX_TRIES)..."
    if curl -sS --fail "$AIM_SERVER_URL" >/dev/null 2>&1; then
        echo "Aim server reachable at $AIM_SERVER_URL."
        break
    else
        echo "Probe to $AIM_SERVER_URL failed. Trying fallback $AIM_SERVER_FALLBACK..."
        if curl -sS --fail "$AIM_SERVER_FALLBACK" >/dev/null 2>&1; then
            echo "Aim server reachable at fallback $AIM_SERVER_FALLBACK."
            break
        fi
        echo "Waiting for Aim server (attempt $i/$MAX_TRIES)..."
        sleep $SLEEP_SECONDS
    fi
    if [ "$i" -eq "$MAX_TRIES" ]; then
        echo "Warning: Aim server did not become reachable after $MAX_TRIES attempts. Proceeding anyway."
    fi
done

# Check if the stack is already registered
if ! zenml stack list | grep -q "aim_stack"; then
    echo "Stack 'aim_stack' not found. Creating..."

    # Register the tracker (retry if necessary)
    set +e
    ATTEMPTS=0
    MAX_ATTEMPTS=5
    while [ $ATTEMPTS -lt $MAX_ATTEMPTS ]; do
        ATTEMPTS=$((ATTEMPTS + 1))
        echo "Attempt $ATTEMPTS to register experiment-tracker using repo $AIM_REPO_URL..."
        if zenml experiment-tracker register aim_tracker \
            --flavor=aim \
            --repo=${AIM_REPO_URL}; then
            echo "Experiment tracker registered using ${AIM_REPO_URL}."
            break
        else
            echo "Failed to register experiment-tracker (attempt $ATTEMPTS). Retrying in 2s..."
            sleep 2
        fi
    done

    # Register the stack (no strict failure here)
    echo "Registering stack..."
    zenml stack register aim_stack \
        -a default \
        -o default \
        -e aim_tracker || echo "Warning: stack registration failed."
    set -e

    echo "Stack registration attempted."
fi

# Activate the stack
zenml stack set aim_stack

exec "$@"