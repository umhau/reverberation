#!/bin/bash

venv=/opt/reverberation/venv

packages=(

    python3
    python3-tkinter
    portaudio-devel 
    xdotool
    xclip

)

pipages=(

    faster-whisper
    pyaudio
    numpy
    
)

[ `whoami` != 'root' ] && echo 'must be root' && exit 

xbps-install -ySu ${packages[@]}

python3 -m venv --clear --upgrade-deps $venv
source $venv/bin/activate
pip3 install ${pipages[@]}

echo 'done'
