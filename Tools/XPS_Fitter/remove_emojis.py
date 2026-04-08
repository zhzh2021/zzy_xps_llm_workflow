#!/usr/bin/env python
"""Remove all emoji characters from XPS_peakfitting_V2.py"""

import re

# Read the file
with open('XPS_peakfitting_V2.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Define emoji patterns to remove
emoji_pattern = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002600-\U000027BF"  # Miscellaneous Symbols
    "\U0000FE0F"             # Variation Selector
    "]+", flags=re.UNICODE
)

# Remove emojis
cleaned_content = emoji_pattern.sub('', content)

# Write back
with open('XPS_peakfitting_V2.py', 'w', encoding='utf-8') as f:
    f.write(cleaned_content)

print("✓ Emoji removal complete")
print(f"Original size: {len(content)} chars")
print(f"Cleaned size: {len(cleaned_content)} chars")
print(f"Removed: {len(content) - len(cleaned_content)} chars")
