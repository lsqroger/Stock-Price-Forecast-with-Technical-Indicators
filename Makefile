install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

lint:
	pylint --disable=R,C app.py

deploy:
	echo "Deploying app"
	eb deploy hello-env

all: install lint 