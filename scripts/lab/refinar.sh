#!/usr/bin/env bash
# refinar.sh — refinamiento por región (segunda pasada por hueso a alta resolución) en SparkLab.
#
#   bash scripts/lab/refinar.sh cache512            caché de alta resolución (512 px), CPU en paralelo
#   bash scripts/lab/refinar.sh prior               máscara previa de cada caso desde las predicciones fuera de fold de y4
#   bash scripts/lab/refinar.sh corrida y7_refine 0 10   afina el y4 del fold 0, predice con dos pasadas y barre el posproceso
#   bash scripts/lab/refinar.sh cola y7_refine 10   los 5 folds en 3 carriles, en segundo plano
#   bash scripts/lab/refinar.sh confirmar y7_refine 10   posproceso consolidado en 5 folds + tabla emparejada
#   bash scripts/lab/refinar.sh estado
#
# Requiere haber corrido antes la confirmación de y4 (checkpoints, λ y predicciones fuera de fold).
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE="${PENGWIN_LAB_DIR:-/workspace/mio/pengwin}"
CACHE="$BASE/cache"
HI="$BASE/cache512"
OOF="$BASE/oof"
LOGS="$BASE/logs"
FOLDS="${FOLDS_FILE:-$REPO/reports/tuning/cv_folds.json}"
PY="${PYTHON:-python3}"
JOBS="${JOBS:-10}"
ORIGEN="${ORIGEN:-y4_fullres_role}"        # modelo del que se parte y cuyas predicciones dan la máscara previa
EP_ORIGEN="${EP_ORIGEN:-20}"
mkdir -p "$HI" "$OOF" "$LOGS"
cd "$REPO" || exit 1

paso() { echo "=== $(date +%H:%M:%S) $*"; }

cmd_cache512() {
  paso "caché de alta resolución en $HI ($JOBS procesos)"
  for k in $(seq 0 $((JOBS - 1))); do
    $PY scripts/build_slice_cache.py --data-dir "$REPO/01_data" --cache-dir "$HI" --image-size 512 --no-extras \
      --shard "$k/$JOBS" > "$LOGS/cache512_$k.log" 2>&1 &
  done
  wait
  echo "casos en alta resolución: $(ls "$HI" | wc -l)"
  grep -l ERROR "$LOGS"/cache512_*.log 2>/dev/null | sed 's/^/con errores: /'
  paso "caché de alta resolución listo"
}

cmd_prior() {
  paso "máscara previa desde $OOF/${ORIGEN}_f{fold}_e${EP_ORIGEN}"
  $PY scripts/build_prior.py --cache-dir "$CACHE" --folds-file "$FOLDS" --oof "$OOF/${ORIGEN}_f{fold}_e${EP_ORIGEN}" | tail -2
  echo "casos con máscara previa: $(ls "$CACHE"/*/prior_boxes.npy 2>/dev/null | wc -l)"
}

cmd_corrida() {
  local cfg="$1" fold="$2" ep="$3" name="${1}_f${2}_e${3}"
  local origen="${ORIGEN}_f${fold}_e${EP_ORIGEN}"
  paso "$name: afinado desde $origen ($ep épocas, dos pasadas, EMA)"
  $PY scripts/train.py --config "configs/tuning/$cfg.yaml" --cache-dir "$CACHE" --hi-cache-dir "$HI" --name "$name" \
    --epochs "$ep" --folds-file "$FOLDS" --fold "$fold" --init-ckpt "checkpoints/$origen/last.pth" \
    --lambdas-file "reports/tuning/y4xul/lambdas_${origen}.json" ${EXTRA_TRAIN:-} || return 1
  paso "$name: predicciones fuera de fold con dos pasadas (EMA)"
  $PY scripts/cv_predict.py --ckpt "checkpoints/$name/last.pth" --cache-dir "$CACHE" --hi-cache-dir "$HI" \
    --folds-file "$FOLDS" --fold "$fold" --out-dir "$OOF/${cfg}_f${fold}_e${ep}" || return 1
  bash scripts/lab/y4xul.sh pp "$cfg" "$fold" "$ep" || return 1         # dos pasadas
  bash scripts/lab/y4xul.sh pp "$cfg" "$fold" "$ep" p1 || return 1      # el mismo modelo, solo la pasada 1
  paso "$name: segunda pasada selectiva (solo donde la pasada 1 sospecha fractura)"
  $PY scripts/cv_predict.py --ckpt "checkpoints/$name/last.pth" --cache-dir "$CACHE" --hi-cache-dir "$HI" \
    --folds-file "$FOLDS" --fold "$fold" --out-dir "$OOF/${cfg}_f${fold}_e${ep}gate" --gate-px "${GATE_PX:-10}" --skip-metrics || return 1
  bash scripts/lab/y4xul.sh pp "$cfg" "$fold" "$ep" gate || return 1
  if [ "$fold" = "0" ] && [ -f "checkpoints/$name/last_raw.pth" ]; then
    paso "$name: mismas dos pasadas sin EMA (solo fold 0, para medir la EMA)"
    $PY scripts/cv_predict.py --ckpt "checkpoints/$name/last_raw.pth" --cache-dir "$CACHE" --hi-cache-dir "$HI" \
      --folds-file "$FOLDS" --fold "$fold" --out-dir "$OOF/${cfg}_f${fold}_e${ep}raw" --skip-metrics || return 1
    bash scripts/lab/y4xul.sh pp "$cfg" "$fold" "$ep" raw || return 1
  fi
  paso "$name: TERMINADO"
}

cmd_final() {
  # el mismo modelo sobre la partición oficial (train -> val); es el que se evalúa en test
  local cfg="$1" ep="$2" name="${1}_final"
  paso "$name: afinado desde ${ORIGEN}_final ($ep épocas)"
  $PY scripts/train.py --config "configs/tuning/$cfg.yaml" --cache-dir "$CACHE" --hi-cache-dir "$HI" --name "$name" \
    --epochs "$ep" --init-ckpt "checkpoints/${ORIGEN}_final/last.pth" \
    --lambdas-file "reports/tuning/y4xul/lambdas_${ORIGEN}_final.json" || return 1
  paso "$name: TERMINADO"
}

cmd_cola() {
  local cfg="$1" ep="$2"
  nohup bash scripts/lab/refinar.sh final "$cfg" "$ep" > "$LOGS/refinar_${cfg}_final.log" 2>&1 &
  echo "modelo final lanzado (pid $!)"
  for carril in "0 3" "1 4" "2"; do
    nohup bash -c "for f in $carril; do bash scripts/lab/refinar.sh corrida $cfg \$f $ep; done" \
      > "$LOGS/refinar_${cfg}_$(echo $carril | tr -d ' ').log" 2>&1 &
    echo "carril folds [$carril] lanzado (pid $!)"
  done
}

cmd_confirmar() {
  local cfg="$1" ep="$2"
  for sfx in "" p1 gate; do
    paso "confirmación: edt sobre ${cfg}${sfx}"
    $PY scripts/tune_postprocess.py --name "y4xul/conf_${cfg}${sfx}_edt" --cache-dir "$CACHE" --oof "$OOF/${cfg}_f{fold}_e${ep}${sfx}" \
      --folds-file "$FOLDS" --jobs "$JOBS" --grid "seed_depth_mm=3,4,5,6;edge_threshold=0.1,0.2,0.3,0.5" \
      --base "method=edt;seed_min_cm3=0.02;edge_weight=5" > /dev/null || return 1
    paso "confirmación: role sobre ${cfg}${sfx}"
    $PY scripts/tune_postprocess.py --name "y4xul/conf_${cfg}${sfx}_role" --cache-dir "$CACHE" --oof "$OOF/${cfg}_f{fold}_e${ep}${sfx}" \
      --folds-file "$FOLDS" --jobs "$JOBS" --grid "role_seed_depth_mm=1,2;role_threshold=0.3,0.5,0.7;role_smooth_mm=0,1,2,3" \
      --base "method=role;edge_threshold=0.2;seed_min_cm3=0.02;edge_weight=5;seed_depth_mm=5" > /dev/null || return 1
  done
  $PY scripts/lab/confirmacion.py
  paso "confirmación: TERMINADO"
}

cmd_estado() {
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/GPU: /'
  for f in "$LOGS"/refinar_*.log; do
    [ -e "$f" ] || continue
    printf '%-26s %s\n' "$(basename "$f" .log)" "$(grep -v '^\s*$' "$f" | tail -1 | cut -c1-150)"
  done
  grep -l Traceback "$LOGS"/refinar_*.log 2>/dev/null | sed 's/^/CON ERROR: /'
  $PY scripts/lab/progreso.py "${2:-y}" 2>/dev/null | cut -c1-160
}

case "${1:-}" in
  cache512)  cmd_cache512 ;;
  prior)     cmd_prior ;;
  corrida)   cmd_corrida "$2" "$3" "$4" ;;
  cola)      cmd_cola "$2" "$3" ;;
  final)     cmd_final "$2" "$3" ;;
  confirmar) cmd_confirmar "$2" "$3" ;;
  estado)    cmd_estado "$@" ;;
  *)         sed -n '2,11p' "${BASH_SOURCE[0]}" ;;
esac
