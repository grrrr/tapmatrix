# Host tests: [tapmatrix~] against Pd's own chain (delwrite~ -> line~ -> delread4~
# -> *~) and against an ideal computation in double precision (ana.py, numpy).
#   make LIBPD_DIR=/path/to/libpd     (a built libpd: libs/libpd.dylib or .so)
# tm.pd: float engine; tmq.pd: q15 engine (on a host: the C kernels), source at -6 dBFS.
LIBPD_DIR ?= ../../libpd
PYTHON ?= python3
CFLAGS = -O2 -I$(LIBPD_DIR)/libpd_wrapper -I$(LIBPD_DIR)/pure-data/src
LIBS = -L$(LIBPD_DIR)/libs -lpd

.PHONY: all
all: run
	@echo "== float engine (tm.pd)"
	DYLD_LIBRARY_PATH=$(LIBPD_DIR)/libs LD_LIBRARY_PATH=$(LIBPD_DIR)/libs ./run tm.pd
	$(PYTHON) ana.py
	@echo "== q15 engine (tmq.pd)"
	DYLD_LIBRARY_PATH=$(LIBPD_DIR)/libs LD_LIBRARY_PATH=$(LIBPD_DIR)/libs ./run tmq.pd
	$(PYTHON) ana.py

run: run.c
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

.PHONY: clean
clean:
	rm -f run out.f32
