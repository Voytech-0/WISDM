#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

zenml integration install aim -y

if ! zenml stack list | grep -q "aim_stack"; then
    echo "Stack 'aim_stack' not found. Creating..."

    zenml experiment-tracker register aim_tracker \
        --flavor=aim \
        --repo=aim://aim-server:43800 \
        --replace || echo "Tracker might already exist"

    zenml stack register aim_stack \
        -a default \
        -o default \
        -e aim_tracker \
        --replace || echo "Stack might already exist"

    echo "Stack registered successfully."
fi

zenml stack set aim_stack

exec "$@"