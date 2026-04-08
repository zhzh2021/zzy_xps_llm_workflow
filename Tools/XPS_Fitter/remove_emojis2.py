#!/usr/bin/env python
"""Remove specific emojis from XPS_peakfitting_V2.py"""

import codecs

# Read the file with UTF-8 encoding
with open('XPS_peakfitting_V2.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Emoji to text mapping
emoji_replacements = {
    '⚠️': '[WARNING]',
    '✅': '[OK]',
    '❌': '[ERROR]',
    '📄': '[FILE]',
    '📊': '[DATA]',
    '🔍': '[SEARCH]',
    '⚙️': '[GEAR]',
    '🎯': '[TARGET]',
    '📁': '[FOLDER]',
    '🎨': '[PLOT]',
    '📚': '[BOOK]',
    '🔬': '[SCIENCE]',
    '📂': '[DIR]',
    '📋': '[LIST]',
    '📐': '[RULE]',
    '📝': '[NOTE]',
    'ℹ️': '[INFO]',
    '→': '->',  # Arrow character
    '–': '-',   # En-dash
    '…': '...',  # Ellipsis
}

# Process each line
cleaned_lines = []
for line in lines:
    new_line = line
    for emoji, replacement in emoji_replacements.items():
        new_line = new_line.replace(emoji, replacement)
    cleaned_lines.append(new_line)

# Write back
with open('XPS_peakfitting_V2.py', 'w', encoding='utf-8') as f:
    f.writelines(cleaned_lines)

print("[OK] Emoji removal complete")
print(f"Processed {len(lines)} lines")
