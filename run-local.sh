#!/usr/bin/env bash

source config.env

export $(xargs <config.env)
./bin/run
