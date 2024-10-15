#!/bin/bash

./run_weekly_distillation_log_extension_single_rag2.sh > log.livis.ext.dis 2>&1 &
./run_weekly_distillation_log_raw_single_rag2.sh > log.livis.raw.dis 2>&1 &
# ./run_weekly_filter_log_raw_single_rag2.sh > log.livis.filter 2>&1 &

./run_weekly_car_distillation_log_extension_single_rag2.sh > log.car.ext.dis 2>&1 &
./run_weekly_car_distillation_log_raw_single_rag2.sh > log.car.raw.dis 2>&1 &
# ./run_weekly_car_filter_log_raw_single_rag2.sh > log.car.filter 2>&1 &