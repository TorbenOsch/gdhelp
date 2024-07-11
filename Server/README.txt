Requirements:
	Python 3.7 oder höher
	Ein finetuned Model and Tokenizer

Required libraries:
	torch
	transformers
	peft
	json
	http.server

Um das Script ausführen zu können, wird der neuste Grafikkartentreiber und Cuda benötigt!
Außerdem wird empfohlen mindestens eine A100 GPU zu benutzen, da das Model sehr groß ist.
Zusätzlich wird mindestens 30GB freier Speicherplatz für das Modell benötigt!

Das Script kann mit folgendem Befehl ausgeführt werden:

1. Server starten und Model laden
	python main.py
