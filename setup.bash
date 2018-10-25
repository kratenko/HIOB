#!/usr/bin/env bash

(
    cd "$(dirname "$0")"
    echo $(pwd)

    setup_hash="$(md5sum setup.bash)"
    echo "Updating sources..."
    git pull
    setup_new_hash="$(md5sum setup.bash)"

    if [[ "$setup_hash" != "$setup_new_hash" ]]
    then
        echo "Setup has been updated, starting new version..."
        ./setup.bash @arg
        exit 0
    fi

    COMPUTE_DEVICE=""
    VENV_PATH=""
    INTERACTIVE=1
    last=""
    for arg in $@
    do
        if [[ "$last" == "--venv" ]]
        then
            VENV_PATH="$arg"
        fi

        if [[ "$arg" == "-i" ]]
        then
            INTERACTIVE=0
            if [[ -z "$COMPUTE_DEVICE" ]]
            then
                COMPUTE_DEVICE=2
            fi
        elif [[ "$arg" == "--gpu" ]]
        then
            COMPUTE_DEVICE=2
        elif [[ "$arg" == "--cpu" ]]
        then
            COMPUTE_DEVICE=1
        fi
        last="$arg"
    done

    MINOR_VER=$(python3 -V | cut -d. -f2)
    if [[ ! $(command -v python) ]] || [[ "$MINOR_VER" -le 5 ]]
    then
        (>&2 echo "No compatible python version found! Ensure that python3 is installed and >= 3.5.")
        exit 1
    fi

    if [[ ! $(command -v virtualenv) ]]
    then
        (>&2 echo "The command 'virtualenv' could not be found! Is it installed?")
        exit 1
    fi

    if [[ -z "$COMPUTE_DEVICE" ]]
    then
        echo "Do you want to run HIOB on the CPU[1] or the GPU[2]?"
        read COMPUTE_DEVICE
        if [[ "$COMPUTE_DEVICE" != "1" ]] && [[ "$COMPUTE_DEVICE" != "2" ]]
        then
            (>&2 echo "Invalid input: Must be '1' or '2'!")
            exit 1
        fi
    fi


    if [[ -z "$VENV_PATH" ]]
    then
        echo "Please enter the path to your venv (or leave empty for './hiob_venv'):"
        read VENV_PATH
        if [[ -z "$VENV_PATH" ]]
        then
            VENV_PATH="./hiob_venv"
        fi
    fi

    if [[ ! -d "$VENV_PATH" ]]
    then
        virtualenv -p python3 "$VENV_PATH"
    elif [[ ! -f "$VENV_PATH/bin/pip" ]]
    then
        (>&2 echo "ERROR: Virtual environment directory exists, but is not a valid virtualenv!")
        exit 1
    fi

    requirements_file="$(realpath ./requirements.txt)"
    if [[ "$COMPUTE_DEVICE" == "1" ]]
    then
        requirements_file="$(realpath ./requirements_cpu.txt)"
    fi

    "$VENV_PATH/bin/pip" install -r "$requirements_file" || {
        (>&2 echo "An error occured. Installation failed")
        exit 1
    }

    echo ""
    echo "All done. HIOB is installed."
    echo "HIOB can now be launched with '$VENV_PATH/bin/python hiob_cli.py' or '$VENV_PATH/bin/python hiob_gui.py'."
)