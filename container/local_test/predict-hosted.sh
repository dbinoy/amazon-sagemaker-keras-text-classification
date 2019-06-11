#!/bin/bash

payload=$1
content=${2:-text/csv}

curl -d @${payload} -H "Content-Type: ${content}" https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/sagemaker-keras-text-classification-2019-06-11-07-33-44-263/invocations
