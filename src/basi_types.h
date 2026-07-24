#ifndef BASI_TYPES_H
#define BASI_TYPES_H

/* One chat message. BASI stores conversations as arrays of these and serializes
   them to OpenAI JSON (basi_messages_to_json) for the server to template. This
   replaces the former use of BasiMsg so BASI needs no llama.h — the
   binary links no libllama and doesn't even compile against its headers. */
typedef struct BasiMsg {
    const char *role;
    const char *content;
} BasiMsg;

#endif /* BASI_TYPES_H */
