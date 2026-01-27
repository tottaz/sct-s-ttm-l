#!/bin/bash

# Exit on error
set -e

echo "Generating macOS icons..."

# Directory containing the .iconset
ICONSET_DIR="assets/icon.iconset"
ICON_PNG="assets/icon.png"

# Check if iconset exists, if so remove it to ensure clean build
if [ -d "$ICONSET_DIR" ]; then
    rm -rf "$ICONSET_DIR"
fi

echo "Creating iconset from $ICON_PNG..."
mkdir -p "$ICONSET_DIR"

# Scale PNG to various sizes
sips -z 16 16     "$ICON_PNG" --out "$ICONSET_DIR/icon_16x16.png" > /dev/null 2>&1
sips -z 32 32     "$ICON_PNG" --out "$ICONSET_DIR/icon_16x16@2x.png" > /dev/null 2>&1
sips -z 32 32     "$ICON_PNG" --out "$ICONSET_DIR/icon_32x32.png" > /dev/null 2>&1
sips -z 64 64     "$ICON_PNG" --out "$ICONSET_DIR/icon_32x32@2x.png" > /dev/null 2>&1
sips -z 128 128   "$ICON_PNG" --out "$ICONSET_DIR/icon_128x128.png" > /dev/null 2>&1
sips -z 256 256   "$ICON_PNG" --out "$ICONSET_DIR/icon_128x128@2x.png" > /dev/null 2>&1
sips -z 256 256   "$ICON_PNG" --out "$ICONSET_DIR/icon_256x256.png" > /dev/null 2>&1
sips -z 512 512   "$ICON_PNG" --out "$ICONSET_DIR/icon_256x256@2x.png" > /dev/null 2>&1
sips -z 512 512   "$ICON_PNG" --out "$ICONSET_DIR/icon_512x512.png" > /dev/null 2>&1
sips -z 1024 1024 "$ICON_PNG" --out "$ICONSET_DIR/icon_512x512@2x.png" > /dev/null 2>&1

# Strip metadata and clear extended attributes which can cause iconutil to fail
for f in "$ICONSET_DIR"/*.png; do
    sips -s format png "$f" --out "$f" > /dev/null 2>&1
    xattr -c "$f" > /dev/null 2>&1
done

# Convert iconset to icns
iconutil -c icns "$ICONSET_DIR" -o assets/icon.icns

echo "icon.icns created in assets/"
