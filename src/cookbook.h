#ifndef BASI_COOKBOOK_H
#define BASI_COOKBOOK_H

/* Model cookbook — discover, download, and manage local GGUF models, ported
 * from odysseus's cookbook feature but rebuilt for a self-contained C CLI that
 * runs models in-process (no serve/tmux). Downloads go straight through curl +
 * HuggingFace resolve URLs into a scanned models dir; /model then switches.
 *
 * `args` is the text after "/cookbook" (leading spaces already trimmed by the
 * caller; may be empty). The caller must have handed the terminal back to
 * cooked mode (raw off, status bar suspended) before calling — get/rm print
 * progress bars and read confirmations on the normal terminal. */
void cookbook_command(const char *args);

#endif /* BASI_COOKBOOK_H */
