import {
  getOpenAIChatRequestParameters,
  getOpenAIModelId,
  isCloudflareAnthropicModel,
  isCloudflareGoogleAIStudioModel,
  isOpenAIGpt5Model,
  isOpenAIReasoningModel,
} from "../app/utils/openai";

describe("OpenAI model helpers", () => {
  test("normalizes provider-prefixed model ids", () => {
    expect(getOpenAIModelId("openai/gpt-5.5")).toBe("gpt-5.5");
    expect(getOpenAIModelId("gpt-5-mini@OpenAI")).toBe("gpt-5-mini");
  });

  test("detects GPT-5 models with provider prefixes", () => {
    expect(isOpenAIGpt5Model("gpt-5")).toBe(true);
    expect(isOpenAIGpt5Model("openai/gpt-5.5")).toBe(true);
    expect(isOpenAIGpt5Model("openai/gpt-4o")).toBe(false);
  });

  test("detects reasoning models with provider prefixes", () => {
    expect(isOpenAIReasoningModel("o3-mini")).toBe(true);
    expect(isOpenAIReasoningModel("openai/o4-mini")).toBe(true);
    expect(isOpenAIReasoningModel("openai/gpt-5.5")).toBe(false);
  });

  test("detects Cloudflare Google AI Studio Gemini models", () => {
    expect(isCloudflareGoogleAIStudioModel("google-ai-studio/gemini-3.5-flash")).toBe(true);
    expect(isCloudflareGoogleAIStudioModel("google/gemini-3.1-pro")).toBe(true);
    expect(isCloudflareGoogleAIStudioModel("openai/gpt-5.5")).toBe(false);
  });

  test("omits unsupported penalty and top_p parameters for Cloudflare Google AI Studio", () => {
    expect(
      getOpenAIChatRequestParameters("google-ai-studio/gemini-3.5-flash", {
        temperature: 0.7,
        presence_penalty: 0,
        frequency_penalty: 0,
        top_p: 1,
      }),
    ).toEqual({ temperature: 0.7 });
  });

  test("keeps standard OpenAI parameters for OpenAI models", () => {
    expect(
      getOpenAIChatRequestParameters("openai/gpt-4o", {
        temperature: 0.7,
        presence_penalty: 0.1,
        frequency_penalty: 0.2,
        top_p: 0.9,
      }),
    ).toEqual({
      temperature: 0.7,
      presence_penalty: 0.1,
      frequency_penalty: 0.2,
      top_p: 0.9,
    });
  });

  test("detects Cloudflare Anthropic models", () => {
    expect(isCloudflareAnthropicModel("anthropic/claude-sonnet-4-6")).toBe(true);
    expect(isCloudflareAnthropicModel("openai/gpt-5.5")).toBe(false);
  });

  test("does not send temperature and top_p together for Cloudflare Anthropic", () => {
    expect(
      getOpenAIChatRequestParameters("anthropic/claude-sonnet-4-6", {
        temperature: 0.5,
        presence_penalty: 0,
        frequency_penalty: 0,
        top_p: 1,
      }),
    ).toEqual({ temperature: 0.5 });

    expect(
      getOpenAIChatRequestParameters("anthropic/claude-sonnet-4-6", {
        temperature: 0.5,
        presence_penalty: 0,
        frequency_penalty: 0,
        top_p: 0.8,
      }),
    ).toEqual({ top_p: 0.8 });
  });

  test("omits deprecated sampling parameters for Cloudflare Anthropic Opus 4.7+", () => {
    expect(
      getOpenAIChatRequestParameters("anthropic/claude-opus-4-8", {
        temperature: 0.5,
        presence_penalty: 0,
        frequency_penalty: 0,
        top_p: 0.8,
      }),
    ).toEqual({});
  });
});
