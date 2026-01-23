#!/bin/bash

# Configuration
APP_NAME="Sattmal"
DMG_NAME="Sattmal_Installer_Silicon.dmg"
VOL_NAME="Sattmal Installer"
SRC_FOLDER="dist/Sattmal.app"
VENV_BIN="./.venv/bin"
SPEC_FILE="sct-sattmal.spec"

# Code Signing Identity (replace with your "Developer ID Application: Your Name (ID)")
SIGNING_IDENTITY="-" # Ad-hoc signing by default

# Exit on error
set -e

echo "🚀 Starting build process for $APP_NAME..."

# 0. Ollama Check
echo "🔍 Checking for Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "⚠️ Ollama is not installed."
    echo "To use 'Explain Mode' AI features, please install Ollama from https://ollama.com/"
    echo "You can continue the build, but the app will prompt for Ollama if AI features are used."
    # Optional: Automatically open Ollama download page
    # open https://ollama.com/download/mac
else
    echo "✅ Ollama is installed."
fi

# 1. Clean up previous builds
echo "🧹 Cleaning up..."
if [ -d "dist" ] || [ -d "build" ]; then
    if ! rm -rf build dist 2>/dev/null; then
        echo "Warning: Standard cleanup failed (Permission denied)."
        echo "Attempting to force cleanup with sudo..."
        sudo rm -rf build dist
    fi
fi
# Also clean any old spec files that might conflict if not using ours
rm -f DocuSignDemo.spec

# 2. Setup Environment & Install Dependencies
echo "📦 Installing dependencies..."
if [ -d ".venv" ]; then
    $VENV_BIN/pip install -r requirements.txt
    $VENV_BIN/pip install pyinstaller pywebview
else
    echo "❌ Virtual environment .venv not found. Please create it first."
    exit 1
fi

# 3. Generate App Icon
echo "🎨 Generating app icon..."
./scripts/build_icons.sh

# 4. Build .app Bundle using PyInstaller (ARM64)
echo "🏗️ Building Silicon (ARM64) .app bundle..."
$VENV_BIN/pyinstaller "$SPEC_FILE" --noconfirm

# 5. Sign the .app bundle (Ad-hoc)
echo "✍️ Signing .app bundle..."
codesign --force --deep --sign "$SIGNING_IDENTITY" "$SRC_FOLDER"

# 6. Build .pkg Installer
echo "📦 Building .pkg installer..."
./scripts/build_pkg.sh

# 7. Create DMG with the .pkg installer
echo "💽 Creating DMG..."

# Create a temporary folder for the DMG content
mkdir -p dist/dmg_content
cp "dist/$APP_NAME-Installer.pkg" dist/dmg_content/

# Create the DMG
hdiutil create \
  -volname "$VOL_NAME" \
  -srcfolder dist/dmg_content \
  -ov -format UDZO \
  "dist/$DMG_NAME"

# 8. Sign DMG
echo "✍️ Signing DMG..."
codesign --force --sign "$SIGNING_IDENTITY" "dist/$DMG_NAME"

# Clean up
rm -rf dist/dmg_content

echo ""
echo "✅ Build complete!"
echo "   App Bundle: $SRC_FOLDER"
echo "   Installer:  dist/$APP_NAME-Installer.pkg"
echo "   DMG:        dist/$DMG_NAME"
echo ""
echo "To distribute:"
echo "   - Share the DMG file for easy distribution"
echo "   - Users will mount the DMG and run the .pkg installer"
