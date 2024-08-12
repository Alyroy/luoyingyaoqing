#!/bin/bash

./run_weekly_distillation_log.sh > run_weekly_distillation_log.log 2>&1 &
./run_weekly_distillation_extand.sh > run_weekly_distillation_extand.log 2>&1 &
