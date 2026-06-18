// C++ shim: render a model's native chat template via llama.cpp's jinja engine
// (libllama-common). The rest of BASI is C; this is the one C++ translation
// unit, exposing a single extern "C" entry point.
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <utility>

#include "llama.h"
#include "chat.h"          // llama.cpp common/chat.h (common_chat_templates_*)
#include "chat_tmpl.h"

extern "C" char *basi_render_chat(const struct llama_model *model,
                                  const struct llama_chat_message *msgs, size_t n_msgs,
                                  bool add_gen_prompt) {
    if (!model || !msgs) return nullptr;
    try {
        // The templates object is parsed once per model and reused (BASI loads
        // a single model and generates single-threaded). `now` is pinned on
        // first use so date-aware templates render a STABLE prefix across turns
        // — otherwise the main loop's delta-prompt bookkeeping would break.
        static const llama_model *cached_model = nullptr;
        static common_chat_templates_ptr tmpls;
        static std::chrono::system_clock::time_point fixed_now =
            std::chrono::system_clock::now();

        if (!tmpls || cached_model != model) {
            tmpls = common_chat_templates_init(model, "");
            cached_model = model;
        }
        if (!tmpls) return nullptr;

        common_chat_templates_inputs in;
        in.use_jinja = true;
        in.add_generation_prompt = add_gen_prompt;
        in.now = fixed_now;
        in.messages.reserve(n_msgs);
        for (size_t i = 0; i < n_msgs; i++) {
            common_chat_msg m;
            m.role    = msgs[i].role    ? msgs[i].role    : "";
            m.content = msgs[i].content ? msgs[i].content : "";
            in.messages.push_back(std::move(m));
        }

        common_chat_params params = common_chat_templates_apply(tmpls.get(), in);
        const std::string &p = params.prompt;
        char *out = static_cast<char *>(malloc(p.size() + 1));
        if (!out) return nullptr;
        memcpy(out, p.data(), p.size());
        out[p.size()] = '\0';
        return out;
    } catch (...) {
        return nullptr;   // any jinja/parse error → caller falls back
    }
}
