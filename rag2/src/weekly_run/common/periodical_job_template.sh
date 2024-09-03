#!/bin/bash
#
# This is the template of each job which should work with crontab to run some
# periodical tasks automatically.
#
# Usage:
#   # 1. Source this template shell script.
#   source periodical_job_template.sh
#
#   # 2. User defines his/her own service logic in one (or several) function(s),
#   # say: ExecuteJobFunction, and defines his/her own logic to check if
#   # dependencies are all ready in one (or several) function(s), say:
#   # CheckDependencyFunction.
#   ...
#
#   # 3. Define the following four variables related with this job.
#   JOB_ROOT_DIR="/home/data/run/periodical_job_example"
#   JOB_NAME="PeriodicalJob.Example"
#   JOB_DEPENDENCY_COMMAND_LIST="CheckDependencyFunction"
#   JOB_COMMAND_LIST="ExecuteJobFunction"
#
#   # 4. Execute job.
#   ExecutePeriodicalJob
#
#   # 5. Add periodical task into crontab.
#
# NOTE: In each self-defined function mentioned above, user should guarantee
#       that the function will invoke exit to terminal the process if the
#       condition it checks does not satisfy itself.

# The root dir path of the running job.
JOB_ROOT_DIR=

# The done file's root dir for jobs whose done file will touch on hdfs.
HDFS_PRODUCTION_DONE_PATH=

# The name of the job.
JOB_NAME=

# The command(s) to check if the dependencies of this job is(are) all ready.
# There could be more than one command seperated by ";".
JOB_DEPENDENCY_COMMAND_LIST=

# The command(s) to do the real task of this job.
# There could be more than one command seperated by ";".
JOB_COMMAND_LIST=

COMMAND_DELIMITER=";"

##############################################
#
# Print log with date.
#
##############################################
function PrintLog() {
  echo -e "[`date "+%Y-%m-%d %H:%M:%S"`] $0:$LINENO " $@
}

##############################################
#
# Check if the job is running.
#
##############################################
function CheckJobIsRunning() {
  PrintLog "Check if the job: ${JOB_NAME} is running."

  local cur_day=`date "+%Y-%m-%d"`
  local pid_path="${JOB_ROOT_DIR}/pid_logs"
  local pid_file="${pid_path}/${JOB_NAME}.pid"

  if [ -e ${pid_file} ]
  then
    local PID=`cat ${pid_file}`
    local script_name=`echo $0 | awk -F"/" '{print $NF}'`
    if [ "`ps aux | grep $script_name | awk '$2'==${PID}'{print "OK"}'`" == "OK" ];then
      PrintLog "Job: ${JOB_NAME} is running."
      exit 1
    fi
    rm -f ${pid_file};
  fi

  local pid=$$
  local dir=`dirname $pid_file`
  mkdir -p ${dir}
  echo $pid > ${pid_file}
}

#############################################
#
# If a job has been done, its pid is expired.
# Other process may use this pid. So we mark
# it as expired.
#
#############################################
function MarkPidExpired() {
  PrintLog "The job has been done, we mark the pid is expired."

  local pid_file="${JOB_ROOT_DIR}/pid_logs/${JOB_NAME}.pid"
  if [ -e ${pid_file} ]
  then
    sed -i 's/^/expired-/g' $pid_file
  fi
}

##############################################
#
# Check if the job should be executed on current hour according to the specified frequency.
#
##############################################
function CheckCurrentHourIsValid() {
  PrintLog "Check if the job should be executed on current hour : ${JOB_NAME}."

  local frequency=1
  local cur_hour=`date +%-H`
  if [ -n "$1" ];then
    if [ "$1" -ge 24 ] || [ "$1" -le 0 ];then
      PrintLog "Frequency is not valid, must greater than 0 and less than 24."
      exit 1
    fi
    frequency=$1
  fi

  if [ ! $(($cur_hour%${frequency})) == 0 ];then
      PrintLog "On current hour shouldn't execute hourly job: ${JOB_NAME}."
      exit 1
  fi
}

##############################################
#
# Check if today's job is done.
#
##############################################
function CheckJobIsDone() {
  PrintLog "Check if today's job: ${JOB_NAME} is done."

  local cur_day=$1
  local done_path="${JOB_ROOT_DIR}/done/${JOB_NAME}"
  local done_file="${done_path}/${cur_day}.done"

  if [ -e $done_file ]
  then
    PrintLog "Today's job: ${JOB_NAME} has already been done."
    exit 1
  fi
}

##############################################
#
# Check if today's job is done for those done
#  file touched in hdfs.
#
##############################################
function CheckHdfsJobIsDone() {
  PrintLog "Check if today's job: ${JOB_NAME} is done."

  local cur_day=$1
  local done_path="${HDFS_PRODUCTION_DONE_PATH}/done/${JOB_NAME}"
  local done_file="${done_path}/${cur_day}/_done"

  timeout -s 9 30s hadoop fs -test -e $done_file
  local res=$?
  if [ $res -eq 124 -o $res -eq 137 ]
  then
    PrintLog "Today's job: ${JOB_NAME} status unknown for timeout, retry later."
    exit 1
  fi

  if [ $res -eq 0 ]
  then
    PrintLog "Today's job: ${JOB_NAME} has already been done."
    exit 1
  fi
}

##############################################
#
# Check if dependencies are all ready.
#
##############################################
function CheckDependenciesAreReady() {
  PrintLog "Check if dependencies of job: ${JOB_NAME} are all ready."

  ExecuteCommands "$JOB_DEPENDENCY_COMMAND_LIST"
}

##############################################
#
# Run this job.
#
##############################################
function RunJob() {
  PrintLog "Run job: ${JOB_NAME}."

  ExecuteCommands "$JOB_COMMAND_LIST"
}

##############################################
#
# Mark that today's job has been done.
#
##############################################
function MarkJobIsDone() {
  PrintLog "Mark that today's job: ${JOB_NAME} has been done."

  local cur_day=$1
  local done_path="${JOB_ROOT_DIR}/done/${JOB_NAME}"
  local done_file="${done_path}/${cur_day}.done"

  if [ ! -d ${done_path} ]
  then
    mkdir -p ${done_path}
  fi

  touch ${done_file}
  if [ $? -ne 0 ]
  then
    PrintLog "Failed to mark today's job: ${JOB_NAME} has been done."
    exit 1
  fi
}

##############################################
#
# Check if today's hourly job is done.
#
##############################################
function CheckHourlyJobIsDone() {
  PrintLog "Check if today's hourly job: ${JOB_NAME} is done."

  local cur_day_hour=$1
  local done_path="${JOB_ROOT_DIR}/done/${JOB_NAME}/${cur_day_hour}"
  local done_file="${done_path}/_done"

  if [ -e $done_file ]
  then
    PrintLog "Today's job: ${JOB_NAME} has already been done."
    exit 1
  fi
}

##############################################
#
# Check if weekly job is done.
#
##############################################
function CheckWeeklyJobIsDone() {
  PrintLog "Check if weekly job: ${JOB_NAME} is done."

  local cur_week=$1
  local done_path="${JOB_ROOT_DIR}/done/${JOB_NAME}"
  local done_file="${done_path}/${cur_week}.done"

  if [ -e $done_file ]
  then
    PrintLog "Weekly job: ${JOB_NAME} has already been done."
    exit 1
  fi
}

##############################################
#
# Mark that hourly's job has been done.
#
##############################################
function MarkHourlyJobIsDone() {
  PrintLog "Mark that today's hourly job: ${JOB_NAME} has been done."

  local cur_day_hour=$1
  local done_path="${JOB_ROOT_DIR}/done/${JOB_NAME}/${cur_day_hour}"
  local done_file="${done_path}/_done"

  if [ ! -d ${done_path} ]
  then
    mkdir -p ${done_path}
  fi

  touch ${done_file}
  if [ $? -ne 0 ]
  then
    PrintLog "Failed to mark today's hourly job: ${JOB_NAME} has been done."
    exit 1
  fi
}

##############################################
#
# Mark that weekly job has been done.
#
##############################################
function MarkWeeklyJobIsDone() {
  PrintLog "Mark that weekly job: ${JOB_NAME} has been done."

  local cur_week=$1
  local done_path="${JOB_ROOT_DIR}/done/${JOB_NAME}"
  local done_file="${done_path}/${cur_week}.done"

  if [ ! -d ${done_path} ]
  then
    mkdir -p ${done_path}
  fi

  touch ${done_file}
  if [ $? -ne 0 ]
  then
    PrintLog "Failed to mark weekly job: ${JOB_NAME} has been done."
    exit 1
  fi
}

##############################################
#
# Mark that today's mapred job has been done.
#
##############################################
function MarkHdfsJobIsDone() {
  PrintLog "Mark that today's job: ${JOB_NAME} has been done."

  local cur_day=$1
  local done_path="${HDFS_PRODUCTION_DONE_PATH}/done/${JOB_NAME}/${cur_day}"
  local done_file="${done_path}/_done"

  if ! hadoop fs -test -e $done_path; then
    hadoop fs -mkdir -p $done_path
  fi

  hadoop fs -touchz $done_file
  if [ $? -ne 0 ]; then
    PrintLog "Failed to mark today's job: ${JOB_NAME} has been done."
    exit 1
  fi
}

##############################################
#
# Execute commands.
#
##############################################
function ExecuteCommands() {
  local command_txt=$1

  local OLD_IFS=$IFS

  IFS=${COMMAND_DELIMITER}
  for command in ${command_txt}
  do
    IFS=${OLD_IFS}
    eval ${command}
    if [ $? -ne 0 ]
    then
      PrintLog "Failed to execute command: ${command}"
      exit 1
    fi
    IFS=${COMMAND_DELIMITER}
  done

  IFS=${OLD_IFS}
}

##############################################
#
# Execute the periodical task.
#
##############################################
function ExecutePeriodicalJob() {
  PrintLog "Start to run the periodical job: ${JOB_NAME}."

  local cur_day=`date +%Y-%m-%d`
  CheckJobIsRunning
  CheckJobIsDone "$cur_day"
  CheckDependenciesAreReady
  RunJob
  MarkJobIsDone "$cur_day"
  MarkPidExpired

  PrintLog "Succeeded to run the periodical job: ${JOB_NAME}."
}

##############################################
#
# Execute weekly periodical task.
#
##############################################
function ExecuteWeeklyPeriodicalJob() {
  PrintLog "Start to run the periodical job: ${JOB_NAME}."

  local cur_week=`date +%Y-%W`
  CheckJobIsRunning
  CheckWeeklyJobIsDone "$cur_week"
  CheckDependenciesAreReady
  RunJob
  MarkWeeklyJobIsDone "$cur_week"
  MarkPidExpired

  PrintLog "Succeeded to run the periodical job: ${JOB_NAME}."
}

##############################################
#
# Execute hourly periodical task.
#
##############################################
function ExecuteHourlyPeriodicalJob() {
  PrintLog "Start to run the hourly periodical job: ${JOB_NAME}."

  local cur_day_hour=$1
  CheckJobIsRunning
  CheckHourlyJobIsDone $cur_day_hour
  CheckDependenciesAreReady
  RunJob
  MarkHourlyJobIsDone $cur_day_hour
  MarkPidExpired

  PrintLog "Succeeded to run the periodical job: ${JOB_NAME}."
}

##############################################
#
# Execute hourly periodical task in specified frequency.
#
##############################################
function ExecuteHourlyOnFrequencyPeriodicalJob() {
  local cur_day_hour=`date +%Y-%m-%d/%H`

  CheckCurrentHourIsValid $1
  ExecuteHourlyPeriodicalJob ${cur_day_hour}
}

##############################################
#
# Execute the periodical task for hdfs jobs.
#
##############################################
function ExecuteHdfsPeriodicalJob() {
  PrintLog "Start to run the periodical job: ${JOB_NAME}."

  local cur_day=`date +"%Y/%m/%d"`
  CheckJobIsRunning
  CheckHdfsJobIsDone "$cur_day"
  CheckDependenciesAreReady
  RunJob
  MarkHdfsJobIsDone "$cur_day"
  MarkPidExpired

  PrintLog "Succeeded to run the periodical job: ${JOB_NAME}."
}
