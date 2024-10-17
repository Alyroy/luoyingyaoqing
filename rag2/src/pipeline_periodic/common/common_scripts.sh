#!/bin/bash
#
# Common configurations and some common functions.
#
# Usage:
#   PRODUCTION_ROOT=""
#   source "$PRODUCTION_ROOT/scripts/periodical_job_template.sh"
#   JOB_ROOT_DIR="your_project_root"
#   source "$JOB_ROOT_DIR/scripts/common_scripts.sh"

# Common variables.
DASH_DATE=`date +%Y-%m-%d`
DASH_DATE_YESTERDAY=`date -d "1 day ago" +%Y-%m-%d`
DASH_DATE_THREE_DAYS_AGO=`date -d "3 days ago" +%Y-%m-%d`
DASH_DATE_WEEK_AGO=`date -d "7 days ago" +%Y-%m-%d`
DASH_DATE_MONTH_AGO=`date -d "1 month ago" +%Y-%m-%d`
SLASH_DATE=`date +%Y/%m/%d`
SLASH_DATE_YESTERDAY=`date -d "1 day ago" +%Y/%m/%d`
SLASH_DATE_THREE_DAYS_AGO=`date -d "3 days ago" +%Y/%m/%d`
SLASH_DATE_WEEK_AGO=`date -d "7 days ago" +%Y/%m/%d`
SLASH_DATE_MONTH_AGO=`date -d "1 month ago" +%Y/%m/%d`

# Common functions.
function CheckStatus() {
  if [ $? -ne 0 ]; then
    PrintLog "$1"
    exit 1
  fi
}

function CheckHdfsFileExist() {
  local filename="$1"
  if ! hadoop fs -test -e $filename; then
    PrintLog "$filename not exist!"
    exit 1
  fi
}

function CheckLocalFileExist() {
  local filename="$1"
  if [ ! -f $filename ]; then
    PrintLog "$filename not exist!"
    exit 1
  fi
}

function ClearHdfsDirIfExist() {
  local hdfs_data_dir="$1"
  local node_depth=`echo $hdfs_data_dir | grep -o "/" | wc -l`
  if [ $node_depth -le 6 ]; then
    PrintLog "node_depth < 7!"
    exit 1
  fi

  if hadoop fs -test -e $hdfs_data_dir; then
    hadoop fs -rm -r $hdfs_data_dir
  fi
}

function MakeHdfsDirIfNotExist() {
  local hdfs_data_dir="$1"
  local node_depth=`echo $hdfs_data_dir | grep -o "/" | wc -l`
  if [ $node_depth -le 6 ]; then
    PrintLog "node_depth < 7!"
    exit 1
  fi

  if ! hadoop fs -test -e $hdfs_data_dir; then
    hadoop fs -mkdir -p $hdfs_data_dir
  fi
}

function MakeLocalDirIfNotExist() {
  local local_dir=$1
  if [ ! -d $local_dir ]; then
    mkdir -p $local_dir
    CheckStatus "Failed to create directory $local_dir."
  fi
}

function CreateLocalFileIfNotExist() {
  local local_file=$1
  if [ ! -e $local_file ]; then
    touch $local_file
    CheckStatus "Failed to create $local_file."
  fi
}

function ClearLocalFileIfExist() {
  local local_file=$1
  local node_depth=`echo $local_file | grep -o "/" | wc -l`
  if [ $node_depth -le 4 ]; then
    PrintLog "Node_depth must bigger than 4!"
    exit 1
  fi

  if [ -e $local_file ]; then
    rm -rf $local_file
    CheckStatus "Failed to remove $local_file."
  fi
}

function CheckFileLines() {
  local check_file=$1
  local min_count=$2
  local message="$3"
  local count=`wc -l $check_file | awk '{print $1;}'`
  if [ $count -lt $min_count ]; then
    PrintLog "Failed to $message due to line count $count < $min_count of file: $check_file."
    exit 1
  fi
}
