all:


bump-upload:
	$(MAKE) bump
	$(MAKE) upload
	
bump:
	bumpversion patch

upload:
	git push --tags
	git push
	rm -f dist/*
	python setup.py sdist
	twine upload dist/*



tests-clean:
	rm -rf out-comptests

tests:
	comptests --nonose duckietown_world_tests

tests-contracts:
	comptests --contracts --nonose duckietown_world_tests

tests-contracts-coverage:
	comptests --contracts --coverage --nonose duckietown_world_tests



branch=$(shell git rev-parse --abbrev-ref HEAD)
#
#tag_rpi=duckietown/rpi-duckietown-shell:$(branch)
#tag_x86=duckietown/duckietown-shell:$(branch)
#
#build: build-rpi build-x86
#
#push: push-rpi push-x86
#
#build-rpi:
#	docker build -t $(tag_rpi) -f Dockerfile.rpi .
#
#build-x86:
#	docker build -t $(tag_x86) -f Dockerfile .
#
#build-x86-no-cache:
#	docker build -t $(tag_x86) -f Dockerfile --no-cache .
#
#push-rpi:
#	docker push $(tag_rpi)
#
#push-x86:
#	docker push $(tag_x86)
#
#test:
#	make -C testing
