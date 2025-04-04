#!/bin/bash

sudo -H DOCKER_BUILDKIT=1 docker build -f ./docker/amd64/Dockerfile -t lalweco/crop_tracker:23.10-humble-py3 ../
