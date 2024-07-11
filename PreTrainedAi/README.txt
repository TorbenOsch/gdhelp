Requirements:
	Mindestens eine A100 GPU
	Neuster Nvidia Grafiktreiber
	Cuda
	Mind. 30Gb freier Speicherplatz

Anleitung:

1. pip install torch==2.3.1
2. pip install scikit-learn==1.3.2
3. pip install peft==0.11.1
4. pip install datasets==2.20.0
5. pip install transformers==4.41.2
6. pip install accelerate==0.31.0
7. pip install bitsandbytes==0.42.0

Sowohl das Trainings als auch das Predict Script können nun mit Python ausgeführt werden.
Das Predict Script dient nur als schnelle Möglichkeit das Modell zu testen nach dem Training. Für Nutzanwendung wird empfohlen den Server zu starten