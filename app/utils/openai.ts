import { isAnthropicSamplingUnsupportedModel } from "./anthropic";

export function getOpenAIModelId(model: string) {
  return model.trim().split("@")[0].split("/").pop()?.toLowerCase() ?? model;
}

export function isOpenAIGpt5Model(model: string) {
  return getOpenAIModelId(model).startsWith("gpt-5");
}

export function isOpenAIResponsesOnlyModel(model: string) {
  return /^gpt-5\.6-(terra|sol)(?:[-.]\d.*)?$/.test(getOpenAIModelId(model));
}

export function buildOpenAIResponsesRequest<T>(
  model: string,
  input: T[],
  stream: boolean | undefined,
  maxOutputTokens: number,
) {
  return {
    model,
    input,
    stream,
    max_output_tokens: maxOutputTokens,
  };
}

export function extractOpenAIResponsesText(response: any): string | undefined {
  if (typeof response?.output_text === "string") return response.output_text;

  const parts = response?.output?.flatMap((item: any) =>
    Array.isArray(item?.content) ? item.content : [],
  );
  if (!parts?.length) return undefined;

  return parts
    .map((part: any) =>
      typeof part?.text === "string"
        ? part.text
        : typeof part?.refusal === "string"
        ? part.refusal
        : "",
    )
    .join("");
}

export function parseOpenAIResponsesSSE(text: string) {
  const event = JSON.parse(text);
  return event?.type === "response.output_text.delta" &&
    typeof event?.delta === "string"
    ? event.delta
    : "";
}

export function isOpenAIReasoningModel(model: string) {
  const modelId = getOpenAIModelId(model);

  return (
    modelId.startsWith("o1") ||
    modelId.startsWith("o3") ||
    modelId.startsWith("o4-mini")
  );
}

export function isCloudflareGoogleAIStudioModel(model: string) {
  const modelId = model.trim().toLowerCase();

  return (
    modelId.startsWith("google-ai-studio/gemini") ||
    modelId.startsWith("google/gemini")
  );
}

export function isCloudflareAnthropicModel(model: string) {
  return model.trim().toLowerCase().startsWith("anthropic/");
}

export function getOpenAIChatRequestParameters(
  model: string,
  modelConfig: {
    temperature?: number;
    presence_penalty?: number;
    frequency_penalty?: number;
    top_p?: number;
  },
) {
  const isReasoningModel = isOpenAIReasoningModel(model);
  const isGpt5 = isOpenAIGpt5Model(model);
  const isCloudflareGoogleAIStudio = isCloudflareGoogleAIStudioModel(model);
  const isCloudflareAnthropic = isCloudflareAnthropicModel(model);

  if (isCloudflareAnthropic) {
    if (isAnthropicSamplingUnsupportedModel(model)) {
      return {};
    }

    // Anthropic models reject requests that specify both `temperature` and
    // `top_p`. NextChat defaults top_p to 1, so prefer temperature unless the
    // user explicitly changes top_p away from its no-op default.
    if (typeof modelConfig.top_p === "number" && modelConfig.top_p !== 1) {
      return { top_p: modelConfig.top_p };
    }

    return { temperature: modelConfig.temperature ?? 1 };
  }

  return {
    temperature:
      !isReasoningModel && !isGpt5 ? modelConfig.temperature ?? 1 : 1,
    ...(!isReasoningModel && !isCloudflareGoogleAIStudio
      ? {
          presence_penalty: modelConfig.presence_penalty ?? 0,
          frequency_penalty: modelConfig.frequency_penalty ?? 0,
        }
      : {}),
    ...(!isCloudflareGoogleAIStudio
      ? { top_p: !isReasoningModel ? modelConfig.top_p ?? 1 : 1 }
      : {}),
  };
}
