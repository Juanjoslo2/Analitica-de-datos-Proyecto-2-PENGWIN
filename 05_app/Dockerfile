# Dashboard PENGWIN en GPU (DGX Spark aarch64 + CUDA 13 o cualquier x86 con NVIDIA).
# Se construye desde la raíz del repo:  docker build -f 05_app/Dockerfile -t pengwin-demo .
FROM nvcr.io/nvidia/pytorch:25.09-py3
RUN pip install --no-cache-dir SimpleITK==2.5.6 PyYAML scikit-learn scikit-image pandas fastapi "uvicorn[standard]" python-multipart
WORKDIR /w
COPY pyproject.toml ./
COPY 03_src ./03_src
COPY configs ./configs
COPY 05_app ./05_app
RUN pip install --no-cache-dir -e . --no-deps
WORKDIR /w/05_app
ENV PYTHONIOENCODING=utf-8 MPLCONFIGDIR=/tmp/mpl
EXPOSE 8780
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8780", "--proxy-headers"]
