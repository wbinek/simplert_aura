FROM jupyter/scipy-notebook:lab-3.2.1
USER root
RUN apt-get update && apt-get install -y gcc
USER jovyan
RUN conda install --yes pythreejs ipysheet rtree pysoundfile\
    && pip install git+https://github.com/SiggiGue/pyfilterbank.git \
    && gcc -shared -fPIC /opt/conda/lib/python3.9/site-packages/pyfilterbank/sosfilt.c -std=c99 -o /opt/conda/lib/python3.9/site-packages/pyfilterbank/sosfilt.so \
    && pip install python-sofa \
    && jupyter labextension install  --no-build @jupyter-widgets/jupyterlab-manager\
    && jupyter labextension install --no-build jupyter-threejs \
    && jupyter lab build
COPY --chown=jovyan . /RayTracer
WORKDIR /RayTracer
USER root
RUN python _CythonSetup.py build_ext --inplace
USER jovyan