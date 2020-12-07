all:
	PYTHONHASHSEED=0;
	python bitflip_experiments.py

clean:
	rm -rf logs
