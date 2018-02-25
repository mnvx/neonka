help:
	@echo "    dev"
	@echo "        Initialise virtualenv with dev requirements"
	@echo "    dev-upgrade"
	@echo "        Upgrade packages"
	@echo "    dev-run"
	@echo "        Run command. Example: make dev-run command="manage.py account"
	@echo "    install"
	@echo "        Install requirements for production"

dev:
	python3 -m venv venv
	LOCALPATH=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST)))) && \
	printf '#!/bin/bash\n%s/venv/bin/pip3 "$$@"' $$LOCALPATH > $$LOCALPATH/pip3 && \
	printf '#!/bin/bash\n%s/venv/bin/python3 "$$@"' $$LOCALPATH > $$LOCALPATH/python3 && \
	chmod +x $$LOCALPATH/pip3 && \
	chmod +x $$LOCALPATH/python3 && \
	$$LOCALPATH/pip3 install --upgrade pip && \
	$$LOCALPATH/pip3 install -r $$LOCALPATH/requirements-dev.txt && \
	$$LOCALPATH/pip3 install -r $$LOCALPATH/requirements.txt

dev-upgrade:
	LOCALPATH=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST)))) && \
	$$LOCALPATH/pip3 install -r requirements.txt --upgrade

dev-run:
	LOCALPATH=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST)))) && \
	PYTHONPATH=$$PYTHONPATH:`realpath $$LOCALPATH` && \
	export PYTHONPATH && \
	$$LOCALPATH/python3 $(command)

install:
	pip3 install -r requirements.txt
