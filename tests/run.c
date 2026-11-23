/* Runs a test patch with libpd for 3 s at 48 kHz and writes the 5 output
 * channels to out.f32 (interleaved float32). [tapmatrix~] is loaded as an
 * external from the repository's top folder, as Pd loads it.
 *   run <patch> */
#include <stdio.h>
#include "z_libpd.h"

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: run <patch>\n"); return 2; }
    libpd_init();
    libpd_add_to_search_path("..");
    libpd_init_audio(0, 5, 48000);
    if (!libpd_openfile(argv[1], ".")) { fprintf(stderr, "can't open %s\n", argv[1]); return 1; }
    libpd_start_message(1);
    libpd_add_float(1);
    libpd_finish_message("pd", "dsp");
    FILE *f = fopen("out.f32", "wb");
    static float out[64 * 5], in[1];
    for (int b = 0; b < 2250; b++) {
        libpd_process_float(1, in, out);
        fwrite(out, sizeof(float), 64 * 5, f);
    }
    fclose(f);
    return 0;
}
