#!/usr/bin/env bash
# switch_model.sh - Interactive model switcher for whisper daemon
# Reads and updates config.toml

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.toml"

# Read config values
eval "$(python3 -c "
from config_loader import load_config
c = load_config('$SCRIPT_DIR/config.toml')
print(f'WHISPER_MODELS_DIR=\"{c.paths.models_dir}\"')
print(f'CURRENT_MODEL=\"{c.model.name}\"')
print(f'CURRENT_MODE=\"{c.daemon.mode}\"')
" 2>/dev/null)"

if [ -z "$WHISPER_MODELS_DIR" ]; then
    notify-send "Whisper" "Could not read config" -u critical -t 2000
    exit 1
fi

# List all available whisper models
list_all_models() {
    cat <<EOF
tiny.en
base.en
base.en-q5_1
base.en-q8_0
small.en
small.en-q5_1
small.en-q8_0
medium.en
medium.en-q5_0
large-v3
large-v3-turbo
large-v3-turbo-q5_0
tiny
base
small
medium
large-v2
EOF
}

# Check if model is downloaded
is_downloaded() {
    local model=$1
    [ -f "$WHISPER_MODELS_DIR/ggml-$model.bin" ]
}

# Build menu items
menu_items=""
while IFS= read -r model; do
    if is_downloaded "$model"; then
        if [ "$model" = "$CURRENT_MODEL" ]; then
            menu_items="${menu_items}● ${model} (active)\n"
        else
            menu_items="${menu_items}✓ ${model}\n"
        fi
    else
        menu_items="${menu_items}  ${model}\n"
    fi
done < <(list_all_models)

# Add separator and mode toggle
menu_items="${menu_items}---\n"

if [ "$CURRENT_MODE" = "server" ]; then
    menu_items="${menu_items}◆ Toggle to CLI mode\n"
else
    menu_items="${menu_items}● Toggle to Server mode\n"
fi

menu_items="${menu_items}Download more models...\n"

# Show wofi menu
selected=$(echo -e "$menu_items" | wofi --dmenu --prompt "Select Whisper Model" --width 450 --height 500 --insensitive)

if [ -z "$selected" ]; then
    exit 0
fi

# Handle toggle mode
if echo "$selected" | grep -q "Toggle to"; then
    exec "$SCRIPT_DIR/toggle_server_mode.sh"
fi

# Handle download
if echo "$selected" | grep -q "Download more models"; then
    foot -e bash -c "cd $WHISPER_MODELS_DIR && ./download-ggml-model.sh; echo ''; echo 'Press Enter to close...'; read"
    exit 0
fi

# Extract model name
selected_model=$(echo "$selected" | sed 's/^[●✓ ]*//' | sed 's/ (active)$//' | xargs)

if [ "$selected_model" = "$CURRENT_MODEL" ]; then
    notify-send "Whisper Model" "Already using $selected_model" -t 2000
    exit 0
fi

# Download if needed
if ! is_downloaded "$selected_model"; then
    notify-send "Whisper Model" "Downloading $selected_model..." -t 3000

    if foot -e bash -c "cd $WHISPER_MODELS_DIR && ./download-ggml-model.sh $selected_model && echo '' && echo 'Download complete! Press Enter to close...' && read"; then
        notify-send "Whisper Model" "Downloaded $selected_model successfully" -t 2000
    else
        notify-send "Whisper Model" "Failed to download $selected_model" -u critical -t 3000
        exit 1
    fi
fi

# Update config.toml with new model
sed -i "s/^name = \".*\"/name = \"$selected_model\"/" "$CONFIG_FILE"

# Restart daemon
systemctl --user daemon-reload
systemctl --user restart whisper.service

sleep 0.5
if systemctl --user is-active --quiet whisper.service; then
    notify-send "Whisper Model" "Switched to $selected_model" -t 2000
else
    notify-send "Whisper Model" "Failed to restart service" -u critical -t 3000
    exit 1
fi
