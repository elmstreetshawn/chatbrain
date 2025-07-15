# chatbrain.spec
# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect all sentence-transformers data files
sentence_transformers_data = collect_data_files('sentence_transformers')
transformers_data = collect_data_files('transformers')

# Collect hidden imports
hidden_imports = [
    'sentence_transformers',
    'sentence_transformers.models',
    'sentence_transformers.models.Transformer',
    'sentence_transformers.models.Pooling',
    'sentence_transformers.models.Normalize',
    'transformers',
    'transformers.models',
    'transformers.models.bert',
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'faiss',
    'faiss._swigfaiss',
    'sklearn',
    'sklearn.ensemble',
    'sklearn.ensemble._forest',
    'sklearn.tree',
    'sklearn.tree._tree',
    'numpy',
    'numpy.core',
    'numpy.core._methods',
    'numpy.lib.format',
    'sqlite3',
    'tkinter',
    'tkinter.ttk',
    'tkinter.scrolledtext',
    'tkinter.messagebox',
    'tkinter.filedialog',
    'json',
    'threading',
    'queue',
    'pathlib',
    'datetime',
    'logging',
    'dataclasses',
    'typing',
    'urllib',
    'urllib.request',
    'zipfile',
    'shutil',
    'platform',
    'subprocess',
    'time',
    'requests'
]

# Additional hidden imports for sentence-transformers
hidden_imports.extend(collect_submodules('sentence_transformers'))
hidden_imports.extend(collect_submodules('transformers'))

block_cipher = None

a = Analysis(
    ['chatbrain.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('README.md', '.'),
        *sentence_transformers_data,
        *transformers_data,
    ],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'pandas',
        'scipy',
        'PIL',
        'cv2',
        'tensorflow',
        'keras',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ChatBrain',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True for debugging
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico' if sys.platform == 'win32' else 'assets/icon.icns',
)

# macOS app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='ChatBrain.app',
        icon='assets/icon.icns',
        bundle_identifier='com.elmstreetshawn.chatbrain',
        info_plist={
            'CFBundleName': 'ChatBrain',
            'CFBundleDisplayName': 'ChatBrain',
            'CFBundleVersion': '1.0.0',
            'CFBundleShortVersionString': '1.0.0',
            'CFBundlePackageType': 'APPL',
            'CFBundleExecutable': 'ChatBrain',
            'NSHighResolutionCapable': True,
            'LSMinimumSystemVersion': '10.14.0',
            'NSRequiresAquaSystemAppearance': False,
        }
    )
