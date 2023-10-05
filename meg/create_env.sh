#!/bin/bash
delete_env()
{
    conda env remove --yes --name $1
}

create_env()
{
    conda env create --quiet --name $1 --file $2
}


if `$(conda info --envs | grep -q $1)`; then
    echo "Found \"$1\" conda enviroment"
    diff -B <(conda env export --name $1 | sed 's/prefix.*//') $2 >/dev/null
    if [ $? -eq 0 ]
    then
        echo "$1 is up to date"
    else
        echo "$1 has changed and will be recreated"
        delete_env $1
        create_env $1 $2
    fi
else
    create_env $1 $2
fi