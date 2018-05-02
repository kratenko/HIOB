#!/usr/bin/env bash

# parse arguments

print_usage() {
    echo "Usage: execute_experiments.sh [OPTIONS] tracker_configs_dir environment_config_file"
    echo ""
    echo "  -h, --help show this message"
    echo "  -b         emit a periodic beep sound when experiments are finished."
}

env_config=""
trackers_config_dir=""
do_beep=false

for arg in $@
do
    if [[ "$arg" == "-h" ]]
    then
        print_usage
        exit 0
    elif [[ "$arg" == "-b" ]]
    then
        do_beep=true
    elif [[ -z "$trackers_config_dir" ]]
    then
        trackers_config_dir="$arg"
    elif [[ -z "$env_config" ]]
    then
        env_config="$arg"
    else
        echo "ERROR: Too many arguments!"
        print_usage
        exit 1
    fi
done

if [[ -z "$env_config" ]] || [[ -z "$trackers_config_dir" ]]
then
    echo "ERROR: Missing arguments!"
    print_usage
    exit 1
fi


log_dir=$(cat "$2" | grep 'log_dir:')
log_dir=${log_dir#log_dir:}
log_dir=$(echo "$log_dir" | sed -e 's/^[[ \t]]*//')


if [ -z $(type -t periodic_beep) ]; then
    periodic_beep() {
        echo "Hold Ctrl + C to quit."
        while true;
        do
            paplay /usr/share/sounds/sound-icons/prompt
	    sleep 0.1
        done
    }
fi

echo "trackers directory is \"$trackers_config_dir\""
echo "environment file is \"$env_config\""
echo "log directory is \"$log_dir\""


#mkdir -p "$log_dir"

for conf in "$trackers_config_dir/tracker"*.yaml; do
    tags="$(cat $conf | grep '#tags:')" && \
    tags=${tags#\#tags: } && \
    echo "executing \"$(basename $conf)\"..." && \
    #python hiob_cli.py -e "$env_config" -t "$conf"
    if ! [[ -z "$tags" ]]
    then
        for log in "$log_dir/hiob-execution-"*
        do
            echo "moving \"$log\" to \"$log_dir/[$tags]_$(basename $log)\""
            #mv "$log" "$log_dir/[$tags]_$(basename $log)"
        done
    fi
    sleep 3
done

if $do_beep
then
    periodic_beep
fi
