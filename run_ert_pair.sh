#!/system/bin/sh
set -eu

CPU_ROOT="/data/local/tmp/test/cpu"
GPU_ROOT="/data/local/tmp/test/gpu"
TMP_DIR="/data/local/tmp/ert_tmp$$"
mkdir -p "$TMP_DIR"; trap 'rm -rf "$TMP_DIR"' EXIT INT TERM

say()  { printf '%s\n' "$*"; }
line() { printf '%-10s %8s %8s %8s (%s)\n' "$1" "$2" "$3" "$4" "$5"; }

stats() {
  f="$1"
  [ ! -s "$f" ] && { echo "0 0 0"; return; }

  read mean var <<EOF
$(awk '{s+=$1; ss+=$1*$1} END{m=s/NR; printf "%.3f %.3f", m, ss/NR-m*m}' "$f")
EOF

  median=$(sort -n "$f" | awk '
    {a[NR]=$1}
    END{
      mid = (NR%2) ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2
      printf "%.3f", mid
    }')

  printf "%s %s %s\n" "$mean" "$var" "$median"
}

run_and_collect() {          # $1=DIR  $2=EXE  $3=OUTFILE
  DIR=$1; EXE=$2; OUT=$3
  (
    cd "$DIR" || exit 1
    FIFO=fifo$$
    mkfifo "$FIFO"
    ./"$EXE" >"$FIFO" 2>&1 &
    PID=$!

    while IFS= read -r line; do
      case "$line" in
        BW:*) set -- $line; echo "$2" >> "$OUT" ;;  
        *)    set -- $line; [ $# -ge 2 ] || continue
              [ "$2" -ge 10 ] && kill "$PID" 2>/dev/null ;;
      esac
    done <"$FIFO"

    rm -f "$FIFO"
    wait "$PID" 2>/dev/null || true
  )
}

run_cpu_only() {
  run_and_collect  "$CPU_ROOT/c$1"  driver1      "$TMP_DIR/cpu.txt"
  read m v med <<EOF
$(stats "$TMP_DIR/cpu.txt")
EOF
  line "CPU$1" "$m" "$v" "$med" "run $1/10"
}

run_gpu_only() {
  run_and_collect  "$GPU_ROOT/cl$1" corun_kernel "$TMP_DIR/gpu.txt"
  read m v med <<EOF
$(stats "$TMP_DIR/gpu.txt")
EOF
  line "GPU$1" "$m" "$v" "$med" "run $1/10"
}

run_pair() {
  ( cd "$CPU_ROOT/c$1"; while true; do ./driver1 >/dev/null 2>&1; done ) &
  CPU_PID=$!

  run_and_collect "$GPU_ROOT/cl$1" corun_kernel "$TMP_DIR/pair.txt"
  kill "$CPU_PID" 2>/dev/null; wait "$CPU_PID" 2>/dev/null || true

  read m v med <<EOF
$(stats "$TMP_DIR/pair.txt")
EOF
  line "PAIR$1" "$m" "$v" "$med" "run $1/10"
}

say "===== ERT BENCH (iter cap 100) ====="
printf '%-10s %8s %8s %8s (%s)\n' "LABEL" "MEAN" "VAR" "MEDIAN" "progress"
echo "----------------------------------------------------"

for i in $(seq 1 10); do
  run_cpu_only "$i"
  run_gpu_only "$i"
done
say "----- independent done -----"

for i in $(seq 1 10); do
  run_pair "$i"
done
