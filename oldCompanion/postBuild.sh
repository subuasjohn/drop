#!/bin/bash

set -e

VENV_PYTHON="$HOME/testEnv/bin/python"
INSTALL_DIR="install"
BUILD_DIR="build"
SRC_FILES=(
  "src/video_receiver/video_receiver/receive_rtsp_gst.py"
  "src/video_receiver/video_receiver/receive_rtsp_cv2.py"
  "src/video_receiver/video_receiver/receive_mp4.py"
  "src/sands/sands/SANDS_Vehicle.py"
  "src/sands/sands/SANDS_Utils.py"
  "src/sands/sands/SANDS_StateMachine.py"
  "src/sands/sands/SANDS_Vars.py"
  "src/sands/sands/control_data.py"
  "src/sands/sands/control_mission.py"
  "src/sands/sands/control_test.py"
  "src/sands/sands/control_node.py"
  "src/sands/sands/control_report.py"
  "src/sands/sands/SANDS_Config.py"
  "src/sands/sands/SANDS_Common.py"
  "src/sands/sands/control_status.py"
  "src/detector/detector/detect.py"
  "src/sands/sands/control_node copy.py"
)

MODE="patch"
if [[ "$1" == "--remove" ]]; then
  MODE="remove"
fi

echo "Shebang mode: $MODE"
echo "Using venv python: $VENV_PYTHON"
echo ""

if [[ "$MODE" == "patch" ]]; then

  ## src
  for srcfile in "${SRC_FILES[@]}"; do
    # echo "$srcfile"
    if [[ -f "$srcfile" ]]; then
      first_line=$(head -n 1 "$srcfile")
      if [[ "$first_line" == "#!"*python* ]]; then
        echo "Updating shebang in $srcfile"
        sed -i "1s|.*|#!$VENV_PYTHON|" "$srcfile"
      else
        echo "Inserting shebang in $srcfile"
        tmpfile=$(mktemp)
        {
          echo "#!$VENV_PYTHON"
          cat "$srcfile"
        } > "$tmpfile"
        mv "$tmpfile" "$srcfile"
        chmod +x "$srcfile"
      fi
    fi
  done

  ## build and installs
  find "$INSTALL_DIR" -type f -executable -print0 | while IFS= read -r -d '' script; do
    if head -n 1 "$script" | grep -qE '^#!.*python'; then
      echo "Patching shebang in $script"
      sed -i "1s|.*|#!$VENV_PYTHON|" "$script"
    fi
  done

  ## files under build
  echo "Finding .py files in $BUILD_DIR..."
  PY_FILES=()
  while IFS= read -r -d ''; do
    PY_FILES+=("$REPLY")
  done < <(find "$BUILD_DIR" -type f -name "*.py" -print0)

  echo "Found ${#PY_FILES[@]} .py files to patch"

  ## update or inserts
  for pyfile in "${PY_FILES[@]}"; do
    first_line=$(head -n 1 "$pyfile")
    if [[ "$first_line" == \#!*python* ]]; then
      echo "Updating shebang in $pyfile"
      sed -i "1s|.*|#!$VENV_PYTHON|" "$pyfile"
    else
      echo "Inserting shebang in $pyfile"
      tmpfile=$(mktemp)
      {
        echo "#!$VENV_PYTHON"
        cat "$pyfile"
      } > "$tmpfile"
      mv "$tmpfile" "$pyfile"
      chmod +x "$pyfile"
    fi
  done

elif [[ "$MODE" == "remove" ]]; then
  ## src
  for srcfile in "${SRC_FILES[@]}"; do
    if [[ -f "$srcfile" ]] && grep -qx "#!$VENV_PYTHON" < <(head -n 1 "$srcfile"); then
      echo "Removing shebang from $srcfile"
      sed -i '1d' "$srcfile"
    fi
  done

  ## build and installs
  echo "Removing shebangs that match: #!$VENV_PYTHON"

  find "$INSTALL_DIR" -type f -executable -print0 | while IFS= read -r -d '' script; do
    if head -n 1 "$script" | grep -qx "#!$VENV_PYTHON"; then
      echo "Removing shebang from $script"
      sed -i '1d' "$script"
    fi
  done

  while IFS= read -r -d '' pyfile; do
    if head -n 1 "$pyfile" | grep -qx "#!$VENV_PYTHON"; then
      echo "Removing shebang from $pyfile"
      sed -i '1d' "$pyfile"
    fi
  done < <(find "$BUILD_DIR" -type f -name "*.py" -print0)
fi

echo "Shebang processing done."

