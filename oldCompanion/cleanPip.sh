#!/bin/bash

#########################################
## Removes anything installed by pip
## User + root + venv
## Work in progress, use with caution
######################################### 

PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

echo "[*] Detected Python version: $PYVER"

# Function to nuke a path if it exists
nuke_path() {
    local path="$1"
    if [ -d "$path" ]; then
        echo "[*] Removing: $path"
        rm -rf "$path"
    else
        echo "[~] Not found: $path"
    fi
}

# ---- USER SITE ----
USER_PIP_PATH="$HOME/.local/lib/python$PYVER/site-packages"
nuke_path "$USER_PIP_PATH"

echo "[*] Clearing user pip cache..."
python3 -m pip cache purge

# ---- SYSTEM SITE ----
SYSTEM_PATHS=(
    "/usr/lib/python$PYVER/site-packages"
    "/usr/local/lib/python$PYVER/site-packages"
)

for path in "${SYSTEM_PATHS[@]}"; do
    if [ "$EUID" -ne 0 ]; then
        echo "[*] Attempting sudo removal of $path"
        sudo rm -rf "$path"
    else
        nuke_path "$path"
    fi
done

echo "[*] Clearing root/system pip cache..."
if [ "$EUID" -ne 0 ]; then
    sudo python3 -m pip cache purge
else
    python3 -m pip cache purge
fi

# ---- VIRTUAL ENVIRONMENT DETECTION ----
if [ -n "$VIRTUAL_ENV" ]; then
    echo "[*] Detected virtualenv at: $VIRTUAL_ENV"
    VENV_SITE="$VIRTUAL_ENV/lib/python$PYVER/site-packages"
    nuke_path "$VENV_SITE"
else
    echo "[~] No virtual environment detected."
fi

echo "[✓] Done. All pip-installed packages removed."
