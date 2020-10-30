all:


bump-upload:
	$(MAKE) bump
	$(MAKE) upload

bump: # v2
	bumpversion patch
	git push --tags
	git push

upload: # v3
	dts build_utils check-not-dirty
	dts build_utils check-tagged
	dt-check-need-upload --package duckietown-world-daffy make upload-do

upload-do:
	rm -f dist/*
	rm -rf src/*.egg-info
	python3 setup.py sdist
	twine upload --skip-existing --verbose dist/*

comptest_package=duckietown_world_tests
out=out-comptests
coverage_dir=out-coverage
coverage_include='*src/duckietown_world*'
coveralls_repo_token=Yp4i32KwzL4s6C76DfjJ3e6NqUgkXsv0X
coverage_run=coverage run

tests-clean:
	rm -rf $(out) $(coverage_dir) .coverage .coverage.*

junit:
	mkdir -p $(out)/junit
	comptests-to-junit $(out)/compmake > $(out)/junit/junit.xml

tests:
	comptests --nonose $(comptest_package)

tests-contracts:
	comptests --contracts --nonose  $(comptest_package)

tests-contracts-coverage:
	$(MAKE) tests-coverage-single-contracts
	$(MAKE) coverage-report
	$(MAKE) coverage-coveralls

tests-coverage:
	$(MAKE) tests-coverage-single-nocontracts
	$(MAKE) coverage-report
	$(MAKE) coverage-coveralls

tests-LFV-coverage:
	mkdir -p artifacts
	$(coverage_run) `which dt-world-draw-log` --filename test-data/LFV.json --output artifacts/LFV

tests-maps-coverage:
	mkdir -p artifacts
	$(coverage_run) `which dt-world-draw-maps` --output artifacts/maps


tests-coverage-single-nocontracts:
	-DISABLE_CONTRACTS=1 comptests -o $(out) --nonose -c "exit"  $(comptest_package)
	-DISABLE_CONTRACTS=1 $(coverage_run)  `which compmake` $(out)  -c "rmake"

tests-coverage-single-contracts:
	-DISABLE_CONTRACTS=1 comptests -o $(out) --nonose -c "exit"  $(comptest_package)
	-DISABLE_CONTRACTS=0 $(coverage_run)  `which compmake` $(out) --contracts -c "rmake"

tests-coverage-parallel-contracts:
	-DISABLE_CONTRACTS=1 comptests -o $(out) --nonose -c "exit" $(package)
	-DISABLE_CONTRACTS=0 $(coverage_run)  `which compmake` $(out) --contracts -c "rparmake"

coverage-report:
	coverage combine
	coverage html -d $(coverage_dir)

coverage-coveralls:
	# without --nogit, coveralls does not find the source code
	COVERALLS_REPO_TOKEN=$(coveralls_repo_token) coveralls


tag_36=duckietown-world-test-python-3.6

build-python-3.6:
	docker build -f Dockerfile.test-python-3.6 -t $(tag_36) .

test-python-3.6: build-python-3.6
	docker run -it $(tag_36)

test-python-3.6-local: build-python-3.6

	docker run -it \
		-v ${DT_ENV_DEVELOPER}/src/duckietown-serialization/src/duckietown_serialization_ds1:/usr/local/lib/python3.6/site-packages/duckietown_serialization_ds1:ro \
		-v ${DT_ENV_DEVELOPER}/src/zuper-utils/src/zuper_json:/usr/local/lib/python3.6/site-packages/zuper_json:ro \
		$(tag_36)


black:
	black -l 110 -t py38 .


export:
	dt-world-export-gltf --map udem1 --out out-udem1
	scp -r out-udem1 @sandy:dev/duckietown-rendering-pyrender/code
