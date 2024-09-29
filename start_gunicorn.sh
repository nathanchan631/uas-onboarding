#!/bin/bash

gunicorn main:app -b 0.0.0.0:8003 --chdir ./src --access-logfile -