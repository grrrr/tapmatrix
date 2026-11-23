# tapmatrix~: a matrix of fractional delays for Pd (README.md)
#   make PDINCLUDEDIR=/path/to/pure-data/src     build the external
#   make install PDLIBDIR=~/Documents/Pd/externals
#   make test                                   host tests against Pd's own chain (tests/)
lib.name = tapmatrix~

class.sources = tapmatrix~.c

datafiles = tapmatrix~-help.pd README.md LICENSE.txt

# Pd >= 0.56 (multichannel signals); single precision
cflags = -DPD_FLOATSIZE=32

PDLIBBUILDER_DIR = pd-lib-builder
include $(PDLIBBUILDER_DIR)/Makefile.pdlibbuilder

.PHONY: test
test:
	$(MAKE) -C tests PDINCLUDEDIR=$(PDINCLUDEDIR)
