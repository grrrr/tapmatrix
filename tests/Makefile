# Host tests: [tapmatrix~] against Pd's own chain (delwrite~ -> line~ -> delread4~
# -> *~) and against an ideal computation in double precision (ana.py, numpy).
#   make LIBPD_DIR=/path/to/libpd     (a built libpd: libs/libpd.dylib or .so)
# tm.pd: float engine; tmfixed.pd: fixed engine (on a host: the C kernels), source at -6 dBFS.
# kerneltest: the SIMD float kernels of this machine (NEON, AVX2) against the scalar ones.
LIBPD_DIR ?= ../../libpd
PYTHON ?= python3
CFLAGS = -O2 -I$(LIBPD_DIR)/libpd_wrapper -I$(LIBPD_DIR)/pure-data/src
LIBS = -L$(LIBPD_DIR)/libs -lpd

.PHONY: all
all: run kerneltest
	@echo "== SIMD kernels vs scalar"
	./kerneltest
	@echo "== float engine (tm.pd)"
	DYLD_LIBRARY_PATH=$(LIBPD_DIR)/libs LD_LIBRARY_PATH=$(LIBPD_DIR)/libs ./run tm.pd
	$(PYTHON) ana.py
	@echo "== fixed engine (tmfixed.pd)"
	DYLD_LIBRARY_PATH=$(LIBPD_DIR)/libs LD_LIBRARY_PATH=$(LIBPD_DIR)/libs ./run tmfixed.pd
	$(PYTHON) ana.py

run: run.c
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

# the SIMD float kernels against the scalar ones (needs only Pd's headers)
PDINCLUDEDIR ?= $(LIBPD_DIR)/pure-data/src
kerneltest: kerneltest.c ../arch/scalar.h ../arch/neon.h ../arch/avx2.h ../arch/fixed.h
	$(CC) -O2 -I$(PDINCLUDEDIR) $< -lm -o $@

.PHONY: clean
clean:
	rm -f run kerneltest out.f32
