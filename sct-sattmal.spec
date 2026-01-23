# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main_desktop.py'],
    pathex=[],
    binaries=[],
    datas=[('templates', 'templates'), ('static', 'static'), ('uploads', 'uploads'), ('config.json', '.')],
    hiddenimports=['flask', 'webview', 'pdfplumber', 'docx', 'markdown'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Sattmal',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Sattmal',
)
app = BUNDLE(
    coll,
    name='Sattmal.app',
    icon='assets/icon.icns',
    bundle_identifier='com.sct.sattmal',
)
