#!/bin/bash

# Configuration
APP_NAME="Sattmal"
PKG_NAME="$APP_NAME-Installer.pkg"
IDENTIFIER="com.sct.sattmal"
VERSION="1.0.0"
INSTALL_LOCATION="/Applications"

echo "Building .pkg installer for $APP_NAME..."

# Build the package
pkgbuild --component "dist/$APP_NAME.app" \
         --identifier "$IDENTIFIER" \
         --version "$VERSION" \
         --install-location "$INSTALL_LOCATION" \
         "dist/$PKG_NAME"

echo "$PKG_NAME created in dist/"
