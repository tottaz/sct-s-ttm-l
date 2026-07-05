#!/bin/bash

# Configuration
APP_NAME="Sattmal"
PKG_NAME="$APP_NAME-Installer.pkg"
IDENTIFIER="com.sct.sattmal"
VERSION=$(python3 -c "import json; d=json.load(open('version.json')); print(d.get('version', '0.0.0'))")
INSTALL_LOCATION="/Applications"

echo "Building .pkg installer for $APP_NAME..."

# Build the package
pkgbuild --component "dist/$APP_NAME.app" \
         --identifier "$IDENTIFIER" \
         --version "$VERSION" \
         --install-location "$INSTALL_LOCATION" \
         "dist/$PKG_NAME"

echo "$PKG_NAME created in dist/"
