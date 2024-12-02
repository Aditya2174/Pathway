#!/bin/sh

# Wait for the server to be ready
until curl --output /dev/null --silent --head http://server:8756/v1/retrieve; do
    echo "Waiting for server to be ready..."
    sleep 15
done

# Start your main application
exec "$@"
