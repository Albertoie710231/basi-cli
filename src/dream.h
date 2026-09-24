#ifndef BASI_DREAM_H
#define BASI_DREAM_H

#ifdef __cplusplus
extern "C" {
#endif

/* `basi-cli sleep ...` — learn reviewed lessons from this project's past
 * sessions into .basi/lessons.md (see dream.cpp). Returns an exit code. */
int dream_cmd(int argc, char **argv);

/* `basi-cli import claude ...` — copy Claude Code memory notes for this project
 * into .basi/knowledge/pinned (git-ignored); feedback notes become lesson
 * proposals for `sleep review`. Returns an exit code. */
int dream_import_claude(int argc, char **argv);

#ifdef __cplusplus
}
#endif

#endif /* BASI_DREAM_H */
