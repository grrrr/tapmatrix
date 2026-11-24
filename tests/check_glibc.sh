#!/bin/sh
# Fails if a Linux binary needs a newer glibc than allowed (built on a new
# distribution, it should still load on older ones):
#   tests/check_glibc.sh <max-version> <binary>...     e.g. 2.28 tapmatrix~.l_amd64
# readelf reads the ELF of any architecture (also cross builds).
max=$1; shift
status=0
for f in "$@"; do
    need=$(readelf --dyn-syms -W "$f" | grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' | sort -uV | tail -1)
    if [ -z "$need" ]; then
        echo "$f: no glibc symbols"
    elif [ "$(printf '%s\n%s\n' "$need" "$max" | sort -V | tail -1)" != "$max" ]; then
        echo "$f: needs glibc $need > $max:"
        readelf --dyn-syms -W "$f" | grep -o "[^ ]*@GLIBC_$need" | sort -u | sed 's/^/  /'
        status=1
    else
        echo "$f: glibc $need <= $max"
    fi
done
exit $status
