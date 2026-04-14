// C bindings for llama.cpp
pub const c = @cImport({
    @cInclude("llama.h");
});

// Re-export commonly used types for convenience
pub const Model = *c.llama_model;
pub const Context = *c.llama_context;
pub const Vocab = *const c.llama_vocab;
pub const Sampler = *c.llama_sampler;
pub const Token = c.llama_token;
pub const Batch = c.llama_batch;

pub const ModelParams = c.llama_model_params;
pub const ContextParams = c.llama_context_params;
pub const SamplerChainParams = c.llama_sampler_chain_params;

// Helper functions
pub fn defaultModelParams() ModelParams {
    return c.llama_model_default_params();
}

pub fn defaultContextParams() ContextParams {
    return c.llama_context_default_params();
}

pub fn defaultSamplerChainParams() SamplerChainParams {
    return c.llama_sampler_chain_default_params();
}
