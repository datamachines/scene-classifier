# Needed SHELL since I'm using zsh
SHELL := /bin/bash

set_gdownpy:
	docker build --tag gdownpl:py -f Dockerfile-gdownpy .

run_gdownpl: set_gdownpy
	@if [ "A{GD_URL}" == "A" ]; then echo "Missing GD_URL, aborting"; exit 1; fi
	@if [ "A{GD_OUTFILENAME" == "A" ]; then echo "Missing GD_OUTFILENAME, aborting"; exit 1; fi
	@docker run --rm -it -v `pwd`:/dmc gdownpl:py gdown --id "${GD_URL}" -O "/dmc/${GD_OUTFILENAME}"

# Important to 'url' otherwise make might throw a multiple targets patterns
get_data:
	GD_URL='1wXY6XUlLoAarwObQRGc8HsyTtqRIweNy' GD_OUTFILENAME="places_data.zip" make run_gdownpl
