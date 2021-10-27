FROM jupyter/scipy-notebook:8c15ba8127e7
RUN conda install --yes pythreejs rtree pysoundfile\
    && pip install ipysheet\
    && pip install git+https://github.com/SiggiGue/pyfilterbank.git \
    && gcc -shared -fPIC /opt/conda/lib/python3.8/site-packages/pyfilterbank/sosfilt.c -std=c99 -o /opt/conda/lib/python3.8/site-packages/pyfilterbank/sosfilt.so \
    && pip install python-sofa \
    && jupyter labextension install  --no-build @jupyter-widgets/jupyterlab-manager\
    && jupyter labextension install --no-build jupyter-threejs \
    && jupyter lab build
COPY --chown=jovyan . /RayTracer
WORKDIR /RayTracer
USER root
RUN python _CythonSetup.py build_ext --inplace
USER jovyan