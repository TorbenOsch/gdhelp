Um das Script ausführen zu können, damit die Berechnungen auf der GPU laufen wird das Tool Anaconda und der neuste Grafiktreiber von Nvidia benötigt benötigt!

Nach erfolgreicher Installation werden folgende Befehle in der Konsole benutzt, um die Umgebung aufzusetzen:

1. Umgebung erstellen
	conda create -n py310 python=3.10
2. Umgebung starten
	conda activate py310
3. Cudatoolkit und cudnn installieren
	conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
4. Tensorflow installieren
	python -m pip install "tensorflow==2.10"
5. Pandas installieren
	pip install pandas

Danach können die Scripte in der Umgebung mit folgenden Befehlen ausgeführt werden:

1. KI Erstellen und Trainieren
	python main.py
2. KI Testen
	python testing_prompts.py