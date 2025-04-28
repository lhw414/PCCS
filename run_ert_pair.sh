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
              [ "$2" -ge 5 ] && kill "$PID" 2>/dev/null ;;
      esac
    done <"$FIFO"

    rm -f "$FIFO"
    wait "$PID" 2>/dev/null || true
  )
}

run_cpu_only() {
  OUT="$TMP_DIR/cpu.txt"
  :> "$OUT"                       # ← 파일 내용 비우기
  run_and_collect "$CPU_ROOT/c$1" driver1 "$OUT"

  read m v med <<EOF
$(stats "$OUT")
EOF
  line "CPU$1" "$m" "$v" "$med" "run $1/10"
}

run_gpu_only() {
  OUT="$TMP_DIR/gpu.txt"
  :> "$OUT"                       # ← 파일 내용 비우기
  run_and_collect "$GPU_ROOT/cl$1" corun_kernel "$OUT"

  read m v med <<EOF
$(stats "$OUT")
EOF
  line "GPU$1" "$m" "$v" "$med" "run $1/10"
}

run_pair(){ CPU=$1 GPU=$2
  OUT="$TMP_DIR/pair_${CPU}_${GPU}.txt"; :>"$OUT"
  ( cd "$CPU_ROOT/c$CPU"; while true;do ./driver1 >/dev/null 2>&1;done ) & CPID=$!
  run_and_collect "$GPU_ROOT/cl$GPU" corun_kernel "$OUT"
  kill "$CPID" 2>/dev/null; wait "$CPID" 2>/dev/null||true
  read m v md<<<"$(stats "$OUT")"
  line "P${CPU}-${GPU}" "$m" "$v" "$md" "pair"; 
}

say "===== ERT BENCH (iter cap 100) ====="
printf '%-10s %8s %8s %8s (%s)\n' "LABEL" "MEAN" "VAR" "MEDIAN" "progress"
echo "----------------------------------------------------"

for i in $(seq 1 10); do
  run_cpu_only "$i"
  run_gpu_only "$i"
done
say "----- independent done -----"

for cpu in $(seq 1 10); do
  for gpu in $(seq 1 10); do
    run_pair "$cpu" "$gpu"
  done
done
