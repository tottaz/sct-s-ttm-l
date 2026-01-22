#!/bin/bash

# Exit on error
set -e

APP_NAME="Sattmal"
MAIN_SCRIPT="main_desktop.py"
DIST_DIR="dist"
BUILD_DIR="build"

echo "🚀 Starting macOS build for $APP_NAME..."

# 1. Clean up
echo "🧹 Cleaning previous builds..."
rm -rf "$DIST_DIR" "$BUILD_DIR" *.spec

# 2. Install dependencies
echo "📦 Installing dependencies from requirements.txt..."
# Use python3 -m pip to ensure we use the correct environment
python3 -m pip install -r requirements.txt

# 3. Build .app with PyInstaller
echo "🏗️ Building .app bundle..."
pyinstaller --noconfirm --windowed --name "$APP_NAME" \
    --add-data "templates:templates" \
    --add-data "static:static" \
    --add-data "uploads:uploads" \
    --add-data "config.json:." \
    --hidden-import "flask" \
    --hidden-import "webview" \
    --hidden-import "pdfplumber" \
    --hidden-import "docx" \
    --hidden-import "markdown" \
    "$MAIN_SCRIPT"

# 4. Create .dmg
echo "💽 Creating .dmg installer..."
DMG_NAME="$APP_NAME-Installer.dmg"
hdiutil create -volname "$APP_NAME" -srcfolder "$DIST_DIR/$APP_NAME.app" -ov -format UDZO "$DIST_DIR/$DMG_NAME"

# 5. Create .pkg
echo "📦 Creating .pkg installer..."
PKG_NAME="$APP_NAME.pkg"
pkgbuild --component "$DIST_DIR/$APP_NAME.app" \
    --install-location "/Applications" \
    "$DIST_DIR/$PKG_NAME"

echo "✅ Build complete! Files available in $DIST_DIR/"
echo "   - $APP_NAME.app"
echo "   - $DMG_NAME"
echo "   - $PKG_NAME"
