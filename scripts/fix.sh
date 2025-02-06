#!/bin/bash

# Path to the processing_emu3.py file
EMU3_FILE="/DATA/disk0/nby/xwl/model/Emu3/emu3/mllm/processing_emu3.py"

# Create backup
cp "$EMU3_FILE" "${EMU3_FILE}.backup"

# Add Union import if not present
sed -i '1i from typing import Union' "$EMU3_FILE"

# Replace type hint syntax
sed -i 's/\([A-Za-z]*\) | \([A-Za-z]*\)/Union[\1, \2]/g' "$EMU3_FILE"

echo "Type hints have been updated in $EMU3_FILE"
echo "Original file backed up as ${EMU3_FILE}.backup"