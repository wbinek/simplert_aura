# Instrukcja ucuchomienia

1. Zainstaluj u skonfiguruj docker
2. W folderze z repozytorium zbuduj obraz docker
    `docker build . --tag simpleaura:0.1`
3. Uruchom jupyterlab
    `docker run -it --rm -p 8888:8888 jupyter/datascience-notebook start.sh jupyter lab`
4. Otwórz wyświetlony link w przeglądarce internetowej