FROM jupyter/scipy-notebook:712576e0e96b
RUN conda install --yes qgrid pythreejs rtree pysoundfile\
    && pip install git+https://github.com/SiggiGue/pyfilterbank.git \
    && gcc -shared -fPIC /opt/conda/lib/python3.8/site-packages/pyfilterbank/sosfilt.c -std=c99 -o /opt/conda/lib/python3.8/site-packages/pyfilterbank/sosfilt.so \
    && jupyter labextension install  --no-build @jupyter-widgets/jupyterlab-manager\
    && jupyter labextension install --no-build qgrid2 \
    && jupyter labextension install --no-build jupyter-threejs \
    && jupyter lab build
COPY . /RayTracer
WORKDIR /RayTracer
USER root
RUN python _CythonSetup.py build_ext --inplace
USER jovyan