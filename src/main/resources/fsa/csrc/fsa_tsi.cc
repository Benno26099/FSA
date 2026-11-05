// AI assistance: edited with Claude (Anthropic; Claude Code, Claude Opus 5/5.5):
// the instruction-preload driver change was written with Claude. Reviewed, run and validated by Ben Colbourne.
// See the thesis's Statement on the Use of Generative AI.
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sstream>
#include <fstream>
#include <iostream>
#include "fsa_tsi.h"


void fsa_tsi_t::fsa_host_thread(void *arg) {
    fsa_tsi_t *tsi = static_cast<fsa_tsi_t*>(arg);
    tsi->run();
}

fsa_tsi_t::fsa_tsi_t(int argc, char** argv) : tsi_t(argc, argv)
{
    fsa_host.init(fsa_host_thread, this);
}

void fsa_tsi_t::run() {
    // load_instructions() activates the FSA (reset()) itself, after preloading
    // the instruction queue, so execTime excludes the slow TSI delivery.
    load_instructions(target_args()[0]);
    while(!should_exit()) {
        int state = fsa_tsi_t::STATE_IDLE;
        while (state != fsa_tsi_t::STATE_DONE) {
            read_chunk(0x8008, sizeof(int), &state);
        }
        htif_exit(0);
    }
    stop();
    while (true) {
        switch_to_target();
    }
}

void fsa_tsi_t::load_instructions(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error(
            "could not open " + path + " (did you misspell it?");
    }

    // Get the file size
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        throw std::runtime_error(
            "Error getting the file size " + path);
    }

    size_t filesize = sb.st_size;
    if (filesize % sizeof(uint32_t) != 0) {
        throw std::runtime_error(
            "FSA instruction file size must be a multiple of 4! size: " + filesize
        );
    }
    // Memory-map the file
    void *mapped = mmap(nullptr, filesize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        throw std::runtime_error(
            "Error reading file " + path);
    }
    close(fd);

    // Preload up to PRELOAD_WORDS instruction words while the FSA is idle (the
    // decoder only dequeues when active), then activate, then stream the rest.
    // Activating first would make execTime count TSI delivery, one word per
    // ~11 cycles in simulation, rather than the accelerator.
    uint32_t *data = static_cast<uint32_t *>(mapped);
    size_t n_words = filesize / sizeof(uint32_t);
    for (size_t i = 0; i < n_words; i++) {
        if (i == PRELOAD_WORDS) {
            reset();
        }
        write_chunk(0x8000, sizeof(uint32_t), data);
        data++;
    }
    if (n_words <= PRELOAD_WORDS) {
        reset();
    }
    if (munmap(mapped, filesize) == -1) {
        throw std::runtime_error("Error unmapping file: " + path);
    }
}


void fsa_tsi_t::reset() {
    uint32_t x = 0xffffffff;
    write_chunk(0x8004, 4, &x);
}
