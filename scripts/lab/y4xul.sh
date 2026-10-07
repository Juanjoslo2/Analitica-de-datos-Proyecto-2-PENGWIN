#!/usr/bin/env bash
# y4xul.sh — pasos del cribado y4xul en el laboratorio (SparkLab / DGX Spark).
#
#   bash scripts/lab/y4xul.sh cache                 caché de cortes + borde 3D (CPU, en paralelo)
#   bash scripts/lab/y4xul.sh oraculo               techo del método role con el GT (CPU)
#   bash scripts/lab/y4xul.sh corrida y0_sintl 0 12 λ → entrenamiento → OOF → posproceso (fold 0, 12 épocas)
#   bash scripts/lab/y4xul.sh lanzar  y0_sintl 0 12 lo mismo en segundo plano (nohup), con log
#   bash scripts/lab/y4xul.sh pp      y0_sintl 0 12 solo el barrido de posproceso sobre el OOF que ya exista
#   bash scripts/lab/y4xul.sh estado                última línea de cada log y uso de GPU
#   bash scripts/lab/y4xul.sh resumen               tabla de resultados (scripts/lab/resumen.py)
#
# Datos y resultados pesados van a $BASE (fuera del repositorio); lo versionable queda en reports/.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE="${PENGWIN_LAB_DIR:-/workspace/mio/pengwin}"
CACHE="$BASE/cache"
OOF="$BASE/oof"
LOGS="$BASE/logs"
FOLDS="${FOLDS_FILE:-$REPO/reports/tuning/cv_folds.json}"
PY="${PYTHON:-python3}"
JOBS="${JOBS:-10}"
mkdir -p "$CACHE" "$OOF" "$LOGS" "$REPO/reports/tuning/y4xul"
cd "$REPO" || exit 1

paso() { echo "=== $(date +%H:%M:%S) $*"; }

cmd_cache() {
  paso "caché de cortes en $CACHE ($JOBS procesos)"
  for k in $(seq 0 $((JOBS - 1))); do
    $PY scripts/build_slice_cache.py --data-dir "$REPO/01_data" --cache-dir "$CACHE" --shard "$k/$JOBS" \
      > "$LOGS/cache_$k.log" 2>&1 &
  done
  wait
  paso "borde de fractura 3D"
  for k in $(seq 0 $((JOBS - 1))); do
    $PY scripts/build_slice_cache.py --cache-dir "$CACHE" --edges-only --shard "$k/$JOBS" \
      > "$LOGS/edges_$k.log" 2>&1 &
  done
  wait
  echo "casos con caché: $(ls "$CACHE" | wc -l) | con borde: $(ls "$CACHE"/*/edge.npy 2>/dev/null | wc -l)"
  grep -l ERROR "$LOGS"/cache_*.log "$LOGS"/edges_*.log 2>/dev/null | sed 's/^/con errores: /'
  paso "caché listo"
}

cmd_oraculo() {
  paso "oráculo del método role"
  $PY scripts/oracle_role.py --cache-dir "$CACHE" --folds-file "$FOLDS" --jobs "$JOBS" ${EXTRA_ORACLE:---only-folds 0} 2>&1 \
    | tee "$LOGS/oraculo.log"
}

cmd_pp() {
  local cfg="$1" fold="$2" ep="$3" name="${1}_f${2}_e${3}"
  paso "posproceso edt sobre $name"
  $PY scripts/tune_postprocess.py --name "y4xul/${name}_edt" --cache-dir "$CACHE" --oof "$OOF/${cfg}_f{fold}_e${ep}" \
    --folds-file "$FOLDS" --only-folds "$fold" --jobs "$JOBS" \
    --grid "seed_depth_mm=3,4,5,6;edge_threshold=0.1,0.2,0.3,0.5" \
    --base "method=edt;seed_min_cm3=0.02;edge_weight=5" || return 1
  if grep -q role3 "configs/tuning/$cfg.yaml"; then
    paso "posproceso role sobre $name"
    $PY scripts/tune_postprocess.py --name "y4xul/${name}_role" --cache-dir "$CACHE" --oof "$OOF/${cfg}_f{fold}_e${ep}" \
      --folds-file "$FOLDS" --only-folds "$fold" --jobs "$JOBS" \
      --grid "role_seed_depth_mm=1,2;role_threshold=0.3,0.5,0.7;role_smooth_mm=0,1,2,3" \
      --base "method=role;edge_threshold=0.2;seed_min_cm3=0.02;edge_weight=5;seed_depth_mm=5" || return 1
  fi
}

cmd_corrida() {
  local cfg="$1" fold="$2" ep="$3" name="${1}_f${2}_e${3}"
  local lam="$REPO/reports/tuning/y4xul/lambdas_${name}.json"
  paso "$name: calibración de λ"
  $PY scripts/calibrate_lambdas.py --config "configs/tuning/$cfg.yaml" --cache-dir "$CACHE" \
    --folds-file "$FOLDS" --fold "$fold" --out "$lam" ${EXTRA_CALIB:-} || return 1
  paso "$name: entrenamiento ($ep épocas)"
  $PY scripts/train.py --config "configs/tuning/$cfg.yaml" --cache-dir "$CACHE" --name "$name" --epochs "$ep" \
    --folds-file "$FOLDS" --fold "$fold" --lambdas-file "$lam" ${EXTRA_TRAIN:-} || return 1
  paso "$name: predicciones fuera de fold (última época)"
  $PY scripts/cv_predict.py --ckpt "checkpoints/$name/last.pth" --cache-dir "$CACHE" --folds-file "$FOLDS" \
    --fold "$fold" --out-dir "$OOF/${cfg}_f${fold}_e${ep}" || return 1
  cmd_pp "$cfg" "$fold" "$ep" || return 1
  paso "$name: TERMINADO"
}

cmd_confirmar() {
  # Posproceso sobre los 5 folds a la vez: una sola combinación por modelo (confirmacion.py la elige)
  local ep="${1:-20}"
  for cfg in y0_sintl y1_fullres y4_fullres_role; do
    paso "confirmación: edt sobre $cfg"
    $PY scripts/tune_postprocess.py --name "y4xul/conf_${cfg}_edt" --cache-dir "$CACHE" --oof "$OOF/${cfg}_f{fold}_e${ep}"       --folds-file "$FOLDS" --jobs "$JOBS" --grid "seed_depth_mm=3,4,5,6;edge_threshold=0.1,0.2,0.3,0.5"       --base "method=edt;seed_min_cm3=0.02;edge_weight=5" > /dev/null || return 1
    if grep -q role3 "configs/tuning/$cfg.yaml"; then
      paso "confirmación: role sobre $cfg"
      $PY scripts/tune_postprocess.py --name "y4xul/conf_${cfg}_role" --cache-dir "$CACHE" --oof "$OOF/${cfg}_f{fold}_e${ep}"         --folds-file "$FOLDS" --jobs "$JOBS" --grid "role_seed_depth_mm=1,2;role_threshold=0.3,0.5,0.7;role_smooth_mm=0,1,2,3"         --base "method=role;edge_threshold=0.2;seed_min_cm3=0.02;edge_weight=5;seed_depth_mm=5" > /dev/null || return 1
    fi
  done
  $PY scripts/lab/confirmacion.py
  paso "confirmación: TERMINADO"
}

cmd_lanzar() {
  local name="${1}_f${2}_e${3}"
  nohup bash "$REPO/scripts/lab/y4xul.sh" corrida "$1" "$2" "$3" > "$LOGS/$name.log" 2>&1 &
  echo "lanzado $name (pid $!) -> $LOGS/$name.log"
}

cmd_estado() {
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/GPU: /'
  echo "procesos de entrenamiento: $(pgrep -fc 'scripts/train.py')"
  for f in "$LOGS"/y*_f*_e*.log; do
    [ -e "$f" ] || continue
    printf '%-28s %s\n' "$(basename "$f" .log)" "$(grep -v '^\s*$' "$f" | tail -1 | cut -c1-150)"
  done
}

case "${1:-}" in
  cache)   cmd_cache ;;
  oraculo) cmd_oraculo ;;
  corrida) cmd_corrida "$2" "$3" "$4" ;;
  lanzar)  cmd_lanzar "$2" "$3" "$4" ;;
  pp)      cmd_pp "$2" "$3" "$4" ;;
  confirmar) cmd_confirmar "${2:-}" ;;
  estado)  cmd_estado ;;
  resumen) $PY scripts/lab/resumen.py "${@:2}" ;;
  *)       sed -n '2,12p' "${BASH_SOURCE[0]}" ;;
esac
