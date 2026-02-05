[app]
# App ki basic detail
title = PoolHelperPro
package.name = poolhelper
package.domain = com.sirajking
source.dir = .
source.main = main.py

# REQUIREMENTS (OpenCV aur Numpy sahi se add kar diye hain)
requirements = python3,kivy==2.2.1,numpy,opencv,pillow,hostpython3

# Display Settings
orientation = landscape
fullscreen = True

# PERMISSIONS (Overlay aur Camera ke liye)
android.permissions = SYSTEM_ALERT_WINDOW,READ_EXTERNAL_STORAGE,FOREGROUND_SERVICE,CAMERA

# ANDROID SETTINGS (Jo aapne mangi thin)
android.api = 31
android.minapi = 21
android.ndk = 25b
android.archs = arm64-v8a
android.accept_sdk_license = True

[buildozer]
log_level = 2
warn_on_root = 1
