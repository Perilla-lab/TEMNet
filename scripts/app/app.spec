# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['app.py','addons.py','config.py','helpers.py','input_pipeline.py','model.py','predict.py'],
             pathex=['/home/gorzy/Documents/UDEL/TEMNet_git/scripts/app','/home/gorzy/Documents/UDEL/TEMNet_git/temnet-env/lib/python3.8/site-packages','/home/gorzy/Documents/UDEL/TEMNet_git/temnet-env/lib64/python3.8/site-packages'],
             binaries=[],
             datas=[],
             hiddenimports=['/home/gorzy/Documents/UDEL/TEMNet_git/scripts/app','tensorflow._api','PIL._tkinter_finder','gi._gobject','pkg_resources.py2_warn','pkg_resources.markers'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='TEMNet',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          icon='/home/gorzy/Documents/UDEL/TEMNet_git/scripts/app/assets/logo.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='TEMNet')
