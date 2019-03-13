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



