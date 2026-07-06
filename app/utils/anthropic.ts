export function isAnthropicSamplingUnsupportedModel(model?: string) {
  const modelId = (model ?? "").trim().toLowerCase();

  // Claude Opus 4.7+ models reject custom sampling parameters such as
  // temperature/top_p/top_k. Leave sampling unset so the provider default is used.
  return /claude-opus-4[-.](?:[7-9]|\d{2,})/.test(modelId);
}

export function getAnthropicSamplingParameters(
  modelConfig: {
    temperature?: number;
    top_p?: number;
  },
  model?: string,
) {
  if (isAnthropicSamplingUnsupportedModel(model)) {
    return {};
  }

  const { temperature, top_p } = modelConfig;

  // Anthropic models reject requests that specify both `temperature` and
  // `top_p`. NextChat defaults `top_p` to 1, so prefer temperature unless the
  // user explicitly changes top_p away from its no-op default.
  if (typeof top_p === "number" && top_p !== 1) {
    return { top_p };
  }

  if (typeof temperature === "number") {
    return { temperature };
  }

  if (typeof top_p === "number") {
    return { top_p };
  }

  return {};
}
