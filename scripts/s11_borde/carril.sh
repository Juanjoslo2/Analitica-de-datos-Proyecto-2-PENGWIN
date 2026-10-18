#!/bin/bash
# Semana 11 · un carril de la cola de trabajos: un contenedor NGC que ejecuta en orden los .sh de
# ~/pengwin_s12_jobs/<carril>/pending (pasan a running y luego a done | failed, cada uno con su .log).
# Monta una copia limpia del repositorio (~/pengwin_s12), separada de la de trabajo de las semanas 10-11.
# Uso: bash scripts/s11_borde/carril.sh A      |  para que termine al vaciarse: touch ~/pengwin_s12_jobs/A/STOP
LANE=$1; D=~/pengwin_s12_jobs/$LANE; mkdir -p $D/pending $D/running $D/done $D/failed
docker rm -f pengwin-s12-$LANE >/dev/null 2>&1 || true
docker run -d --rm --name pengwin-s12-$LANE --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/pengwin_s12:/w -v ~/pengwin_data:/data -v $D:/jobs -w /w -e HOME=/w -e MPLCONFIGDIR=/w/.mpl -e PYTHONIOENCODING=utf-8 \
  nvcr.io/nvidia/pytorch:25.09-py3 bash -lc "pip install -q SimpleITK==2.5.6 PyYAML scikit-learn scikit-image==0.25.2 pandas >/dev/null 2>&1; pip install -q -e . --no-deps >/dev/null 2>&1;
  while true; do j=\$(ls /jobs/pending/*.sh 2>/dev/null | sort | head -1);
    if [ -z \"\$j\" ]; then [ -f /jobs/STOP ] && exit 0; sleep 30; continue; fi;
    b=\$(basename \$j); mv \$j /jobs/running/\$b;
    if bash /jobs/running/\$b > /jobs/running/\$b.log 2>&1; then mv /jobs/running/\$b /jobs/running/\$b.log /jobs/done/; else mv /jobs/running/\$b /jobs/running/\$b.log /jobs/failed/; fi;
  done" >/dev/null
echo "carril $LANE arriba (pengwin-s12-$LANE)"
