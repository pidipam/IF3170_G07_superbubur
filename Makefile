setup:
	pip install -r requirements.txt

run:
	jupyter nbconvert --to notebook --execute notebook.ipynb --output output/notebook_executed.ipynb

clean:
	rm -rf output/* __pycache__ src/__pycache__