## How to run corun kernel in NPU

### 1. Build corun kernel

```bash
hexagon-clang -mv68 -O2 -G0 -shared -fPIC -mhvx -mhvx-length=128B -I$HEXAGON_SDK_ROOT/tools/idl -I$HEXAGON_SDK_ROOT/incs -I$HEXAGON_SDK_ROOT/incs/stddef corunkernel.c -o corunkernel.so
```

### 2. Move the so file and other binary files to the android device

```bash
adb push corunkernel.so /data/local/tmp/
adb push $HEXAGON_SDK_ROOT/libs/run_main_on_hexagon/ship/android_aarch64/run_main_on_hexagon /data/local/tmp/
adb push $HEXAGON_SDK_ROOT/libs/run_main_on_hexagon/ship/android_aarch64/hexagon_toolv*_v*/librun_main_on_hexagon_skel.so /data/local/tmp/lib/dsp/
```

### 3. Run the corun kernel and check the log

```bash
adb shell "echo 0x1f > /data/local/tmp/lib/dsp/run_main_on_hexagon.farf"
adb logcat -c
adb shell "export DSP_LIBRARY_PATH=/data/local/tmp/lib/dsp; \
           cd /data/local/tmp && ./run_main_on_hexagon 3 corunkernel.so
adb logcat -v brief -s adsprpc
```