#!/bin/bash
# Semana 12 · orquestador en ANTON. Corre solo en tmux (el PC puede estar apagado):
#   etapa 1  cribado: fold 0, 12 épocas, 4 corridas a la vez (control + c11 / c12 / c13)
#   etapa 2  confirmación de los que pasen (máx. 2): 5 folds × 20 épocas, TTA 5 + edt_hmax fijo
#   etapa 3  modelo final del ganador (40 épocas, split de siempre) y UNA medición de test
# Las reglas están en reports/tuning/s12/PREREGISTRO.md y las aplica scripts/s12/decidir.py.
# Cada etapa (y un latido cada 30 min) se publica en la rama s12-spark del repo privado. Una tarea
# programada en la nube (routine de claude.ai) revisa esa rama y, cuando ETAPA dice "terminado",
# redacta los resultados según reports/tuning/s12/TAREA_NUBE.md.
#
# Uso:   tmux new -d -s pengwin-s12 'bash ~/pengwin_s12/scripts/s12/orquestar.sh'
#        DRY=1 → prueba de humo (1 época de 15 pasos, fold 0, validación en vez de test, sin publicar)
set -u
R=~/pengwin_s12; J=~/pengwin_s12_jobs; S=$R/reports/tuning/s12; T=$R/reports/tuning
DRY=${DRY:-0}
cd $R
mkdir -p $S $J
CANDIDATOS="c11_wd1e2 c12_segw32 c13_unetpp"
if [ "$DRY" = 1 ]; then
  E1=1; E2=1; E3=1; MS="--max-steps 15"; IT=10; FOLDS2="0"; ONLY2="--only-folds 0"; SPLIT3=val
  GRID1='seed_depth_mm=5;edge_threshold=0.2'
else
  E1=12; E2=20; E3=40; MS=""; IT=200; FOLDS2="0 1 2 3 4"; ONLY2=""; SPLIT3=test
  GRID1='seed_depth_mm=1.5,2,3,4,5,6;edge_threshold=0.1,0.2,0.3,0.5'
fi
FILT='{ grep -v -i warn || true; }'

log() { echo "$(date '+%F %T') $*" | tee -a $S/bitacora.log; }
etapa() { echo "$*" > $S/ETAPA; log "== $*"; }

publicar() {                     # $1 = mensaje del commit
  mkdir -p $S/resultados $S/historiales $S/lambdas $S/logs
  cp -f $T/pp_scr_s12_* $T/pp_s12_* $T/borde_s12_* $S/resultados/ 2>/dev/null
  cp -f $R/reports/eval/s12_* $R/reports/fragments/s12_* $S/resultados/ 2>/dev/null
  cp -f $R/reports/train/s12_* $S/historiales/ 2>/dev/null
  cp -f $T/lambdas/s12_* $S/lambdas/ 2>/dev/null
  for f in $J/*/*/*.log; do [ -f "$f" ] && tail -n 80 "$f" > "$S/logs/$(basename "$f")"; done
  python3 $R/scripts/s12/estado.py > $S/ESTADO.md 2>&1
  [ "$DRY" = 1 ] && return
  git -C $R add -A reports/tuning/s12 >/dev/null 2>&1
  git -C $R commit -q -m "s12 (ANTON): $1" >/dev/null 2>&1 || true
  for i in 1 2 3; do git -C $R push -q origin HEAD:s12-spark >>$S/bitacora.log 2>&1 && return; sleep 60; done
  log "no se pudo publicar en GitHub (se reintenta en el próximo latido)"
}

ULTIMO=$(date +%s)
latido() {
  local ahora; ahora=$(date +%s)
  if [ $((ahora - ULTIMO)) -ge 1800 ]; then publicar "latido $(date '+%H:%M')"; ULTIMO=$ahora; fi
}

vigilar_carriles() {             # un carril caído deja su trabajo en running/ para siempre: se marca y se reinicia
  local L f
  for L in A B C D; do
    docker ps --format '{{.Names}}' | grep -qx pengwin-s12-$L && continue
    for f in $J/$L/running/*.sh; do
      [ -f "$f" ] || continue
      mv "$f" $J/$L/failed/ 2>/dev/null; mv "$f.log" $J/$L/failed/ 2>/dev/null
      log "el carril $L se cayó con $(basename $f) en curso: queda como fallido"
    done
    if ls $J/$L/pending/*.sh >/dev/null 2>&1 && [ ! -f $J/$L/STOP ]; then
      bash $R/scripts/s12/carril.sh $L >>$S/bitacora.log 2>&1; log "carril $L reiniciado"
    fi
  done
}

hecho() { ls $J/*/done/$1.sh >/dev/null 2>&1; }
fallo() { ls $J/*/failed/$1.sh >/dev/null 2>&1; }
esperar() {                      # espera a que todos los trabajos nombrados terminen (bien o mal)
  while true; do
    local falta=0
    for b in "$@"; do hecho $b || fallo $b || falta=1; done
    [ $falta = 0 ] && return
    vigilar_carriles; latido; sleep 120
  done
}
encolar() { cat > $J/$1/pending/$2.sh; log "encolado $2 en el carril $1"; }

# ------------------------------------------------------------------ trabajos
trabajo_cribado() {              # $1 config, $2 corrida
cat <<EOF
set -e -o pipefail
FF="--folds-file reports/tuning/cv_folds.json --fold 0"
python -u scripts/calibrate_lambdas.py --config configs/tuning/$1.yaml --cache-dir /data/data_processed \$FF --iters $IT --out reports/tuning/lambdas/$2.json 2>&1 | $FILT | tail -4
python -u scripts/train.py --config configs/tuning/$1.yaml --cache-dir /data/data_processed --name $2 \$FF --epochs $E1 $MS --lambdas-file reports/tuning/lambdas/$2.json 2>&1 | $FILT
python -u scripts/cv_predict.py --ckpt checkpoints/$2/last.pth --cache-dir /data/data_processed \$FF --out-dir /data/oof/$2 2>&1 | $FILT
python -u scripts/s12/borde_pr.py $2 0 | tail -12
export CUDA_VISIBLE_DEVICES=
python -u scripts/tune_postprocess.py --name pp_scr_$2 --cache-dir /data/data_processed --oof /data/oof/$2 \
  --folds-file reports/tuning/cv_folds.json --jobs 5 --only-folds 0 --grid '$GRID1' \
  --base 'method=edt;seed_min_cm3=0.02;edge_weight=5' 2>&1 | $FILT | tail -8
EOF
}

trabajo_conf() {                 # $1 config, $2 fold
local N=s12_conf_$1_f$2
cat <<EOF
set -e -o pipefail
FF="--folds-file reports/tuning/cv_folds.json --fold $2"
python -u scripts/calibrate_lambdas.py --config configs/tuning/$1.yaml --cache-dir /data/data_processed \$FF --iters $IT --out reports/tuning/lambdas/$N.json 2>&1 | $FILT | tail -4
python -u scripts/train.py --config configs/tuning/$1.yaml --cache-dir /data/data_processed --name $N \$FF --epochs $E2 $MS --lambdas-file reports/tuning/lambdas/$N.json 2>&1 | $FILT
python -u scripts/cv_predict.py --ckpt checkpoints/$N/last.pth --cache-dir /data/data_processed \$FF --out-dir /data/oof/$N --tta --skip-metrics 2>&1 | $FILT
EOF
}

trabajo_pp_conf() {              # $1 config
cat <<EOF
set -e -o pipefail
export CUDA_VISIBLE_DEVICES=
python -u scripts/tune_postprocess.py --name pp_s12_conf_$1 --cache-dir /data/data_processed --oof "/data/oof/s12_conf_$1_f{fold}" \
  --folds-file reports/tuning/cv_folds.json --jobs 6 $ONLY2 --grid "seed_depth_mm=5" \
  --base "method=edt_hmax;edge_threshold=0.1;hmax_h_mm=1.5;seed_min_cm3=0.02;edge_weight=5" 2>&1 | $FILT | tail -8
EOF
}

trabajo_final() {                # $1 config
local N=s12_final_$1
cat <<EOF
set -e -o pipefail
python -u scripts/calibrate_lambdas.py --config configs/tuning/$1.yaml --cache-dir /data/data_processed --iters $IT --out reports/tuning/lambdas/$N.json 2>&1 | $FILT | tail -4
python -u scripts/train.py --config configs/tuning/$1.yaml --cache-dir /data/data_processed --name $N --epochs $E3 $MS --lambdas-file reports/tuning/lambdas/$N.json 2>&1 | $FILT
pip install -q pandas >/dev/null 2>&1 || true
python -u scripts/evaluate.py --ckpt checkpoints/$N/last.pth --cache-dir /data/data_processed --split $SPLIT3 2>&1 | $FILT
python -u scripts/evaluate_fragments.py --ckpt checkpoints/$N/last.pth --cache-dir /data/data_processed --data-dir /data/01_data \
  --split $SPLIT3 --config configs/tuned.yaml --tag _tuned 2>&1 | $FILT | tail -25
EOF
}

# ------------------------------------------------------------------ arranque
log "orquestador arriba (DRY=$DRY) en $(git -C $R rev-parse --short HEAD)"
for L in A B C D; do bash $R/scripts/s12/carril.sh $L >>$S/bitacora.log 2>&1; done

# ------------------------------------------------------------------ etapa 1: cribado
etapa "1 · cribado (fold 0, $E1 épocas)"
trabajo_cribado c13_unetpp s12_c13_unetpp_e12_f0 | encolar A 110_scr_c13_unetpp
trabajo_cribado c00_base   s12_ref_e12_f0        | encolar B 110_scr_ref
trabajo_cribado c11_wd1e2  s12_c11_wd1e2_e12_f0  | encolar C 110_scr_c11_wd1e2
trabajo_cribado c12_segw32 s12_c12_segw32_e12_f0 | encolar D 110_scr_c12_segw32
publicar "etapa 1 encolada"
esperar 110_scr_c13_unetpp 110_scr_ref 110_scr_c11_wd1e2 110_scr_c12_segw32
CONF=$(python3 $R/scripts/s12/decidir.py etapa1 2>>$S/bitacora.log | tail -1)
[ "$DRY" = 1 ] && CONF="c13_unetpp"          # la prueba de humo recorre las tres etapas con el más complejo
log "etapa 1 decidida: a confirmar = '${CONF:-ninguno}'"
publicar "etapa 1 terminada: a confirmar = ${CONF:-ninguno}"

# ------------------------------------------------------------------ etapa 2: confirmación
GANADOR=""
if [ -n "$CONF" ]; then
  etapa "2 · confirmación ($CONF; folds $FOLDS2; $E2 épocas)"
  i=0; NOMBRES=""
  for c in $CONF; do for k in $FOLDS2; do
    L=$(echo A B C D | cut -d' ' -f$((i % 4 + 1))); i=$((i + 1))
    trabajo_conf $c $k | encolar $L 2${i}0_conf_${c}_f$k
    NOMBRES="$NOMBRES 2${i}0_conf_${c}_f$k"
  done; done
  publicar "etapa 2 encolada"
  esperar $NOMBRES
  j=0; PPN=""
  for c in $CONF; do
    L=$(echo A B C D | cut -d' ' -f$((j % 4 + 1))); j=$((j + 1))
    trabajo_pp_conf $c | encolar $L 29${j}_pp_conf_$c
    PPN="$PPN 29${j}_pp_conf_$c"
  done
  esperar $PPN
  GANADOR=$(python3 $R/scripts/s12/decidir.py etapa2 2>>$S/bitacora.log | tail -1)
  [ "$DRY" = 1 ] && GANADOR="c13_unetpp"
  log "etapa 2 decidida: ganador = '${GANADOR:-ninguno}'"
  publicar "etapa 2 terminada: ganador = ${GANADOR:-ninguno}"
fi

# ------------------------------------------------------------------ etapa 3: modelo final + test
if [ -n "$GANADOR" ]; then
  etapa "3 · modelo final $GANADOR ($E3 épocas) y test ($SPLIT3)"
  trabajo_final $GANADOR | encolar A 300_final_$GANADOR
  publicar "etapa 3 encolada"
  esperar 300_final_$GANADOR
  publicar "etapa 3 terminada"
fi

# ------------------------------------------------------------------ cierre
for L in A B C D; do touch $J/$L/STOP; done
etapa "terminado (confirmar: ${CONF:-ninguno} · ganador: ${GANADOR:-ninguno})"
publicar "fin de la semana 12 en ANTON"
log "orquestador terminado"
