#!/bin/bash

if [ -z "$PORT" ]; then
  PORT=8501
fi

streamlit run dashboard.py --server.port $PORT
