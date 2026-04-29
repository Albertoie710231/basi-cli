#ifndef BASI_SCAFFOLD_H
#define BASI_SCAFFOLD_H

/* scaffold tool — materialize a template tree into a destination directory. */
char *execute_scaffold(const char *args);

/* Build a printable index of available templates for the system prompt.
 * Returns malloc'd string. */
char *build_templates_index(void);

#endif /* BASI_SCAFFOLD_H */
