[app]
title = PoolHelperPro
package.name = poolhelper
package.domain = com.sirajking
source.dir = .
source.main = main.py
requirements = python3,kivy==2.2.1,opencv-python-headless,numpy,pillow
orientation = landscape
fullscreen = True
android.permissions = SYSTEM_ALERT_WINDOW,READ_EXTERNAL_STORAGE,FOREGROUND_SERVICE,CAMERA
android.archs = armeabi-v7a,arm64-v8a

[buildozer]
log_level = 2
warn_on_root = 1
