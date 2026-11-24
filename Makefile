# tapmatrix~: a matrix of fractional delays for Pd (README.md)
#   make PDINCLUDEDIR=/path/to/pure-data/src     build the external
#   make ... floatsize=64 extension=linux-amd64-64.so   double precision (Pd64)
#   make install PDLIBDIR=~/Documents/Pd/externals
#   make test                                   host tests against Pd's own chain (tests/)
lib.name = tapmatrix~

class.sources = tapmatrix~.c

# the vanilla fallback: abstractions with the same interface. Installed next to
# the binary: in one folder Pd tries the binary first, so the abstraction only
# runs where no binary loads.
datafiles = tapmatrix~-help.pd README.md LICENSE.txt \
	tapmatrix~.pd tapmatrix.in~.pd tapmatrix.out~.pd tapmatrix.tap~.pd

# Pd >= 0.56 (multichannel signals); float or double (floatsize=64)

PDLIBBUILDER_DIR = pd-lib-builder
include $(PDLIBBUILDER_DIR)/Makefile.pdlibbuilder

# the kernels are headers (arch/): rebuild when they change
tapmatrix~.$(extension).o: $(wildcard arch/*.h)

.PHONY: test
test:
	$(MAKE) -C tests PDINCLUDEDIR=$(PDINCLUDEDIR)
