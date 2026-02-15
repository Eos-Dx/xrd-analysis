#!/bin/bash
echo "Stopping Omniscan Orchestrator..."
pkill -f "uvicorn omniscan_orchestrator.rest_server"
echo "Done"
