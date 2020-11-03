FROM jupyter/scipy-notebook
RUN conda install --yes qgrid pythreejs rtree pysoundfile\
    && pip install git+https://github.com/SiggiGue/pyfilterbank.git \
    && jupyter labextension install  --no-build @jupyter-widgets/jupyterlab-manager\
    && jupyter labextension install --no-build qgrid2 \
    && jupyter labextension install --no-build jupyter-threejs \
    && jupyter lab build
COPY . /RayTracer
WORKDIR /RayTracer
USER root
RUN python _CythonSetup.py build_ext --inplace
USER jovyan