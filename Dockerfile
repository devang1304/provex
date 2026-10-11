# PROVEX v2 runtime on CPU (all replays run on CPU; CUDA scatter tie-breaking is nondeterministic).
#   docker build -t provex:2 .
#   docker run --rm provex:2                       # test suite
#   docker run --rm -v "$PWD/artifact:/provex/artifact" -v "$PWD/results:/provex/results" provex:2 \
#       bash scripts/run_experiments.sh <stage>     # experiments (data and models mounted)
FROM python:3.10-slim
WORKDIR /provex
COPY requirements-v2.txt .
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.5.0+cpu.html \
 && pip install --no-cache-dir -r requirements-v2.txt
COPY . .
CMD ["python", "-m", "pytest", "-q", "tests"]