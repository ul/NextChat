import {
  buildOpenAIResponsesRequest,
  extractOpenAIResponsesText,
  getOpenAIChatRequestParameters,
  getOpenAIModelId,
  isCloudflareAnthropicModel,
  isCloudflareGoogleAIStudioModel,
  isOpenAIGpt5Model,
  isOpenAIReasoningModel,
  isOpenAIResponsesOnlyModel,
  parseOpenAIResponsesSSE,
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

  test("detects only Terra and Sol Responses model aliases", () => {
    expect(isOpenAIResponsesOnlyModel("openai/gpt-5.6-terra")).toBe(true);
    expect(isOpenAIResponsesOnlyModel("gpt-5.6-sol-2026-07-01")).toBe(true);
    expect(isOpenAIResponsesOnlyModel("openai/gpt-5.6-terra.1")).toBe(true);
    expect(isOpenAIResponsesOnlyModel("openai/gpt-5.6-pro")).toBe(false);
    expect(isOpenAIResponsesOnlyModel("openai/gpt-5.6-terrestrial")).toBe(
      false,
    );
  });

  test("builds a Responses request without Chat Completions parameters", () => {
    const input = [{ role: "user", content: "hello" }];
    expect(
      buildOpenAIResponsesRequest("openai/gpt-5.6-terra", input, true, 2048),
    ).toEqual({
      model: "openai/gpt-5.6-terra",
      input,
      stream: true,
      max_output_tokens: 2048,
    });
  });

  test("extracts Responses output text and refusals", () => {
    expect(extractOpenAIResponsesText({ output_text: "direct" })).toBe(
      "direct",
    );
    expect(
      extractOpenAIResponsesText({
        output: [
          { content: [{ type: "output_text", text: "hello " }] },
          { content: [{ type: "refusal", refusal: "cannot comply" }] },
        ],
      }),
    ).toBe("hello cannot comply");
  });

  test("parses only Responses output text delta events", () => {
    expect(
      parseOpenAIResponsesSSE(
        JSON.stringify({ type: "response.output_text.delta", delta: "hi" }),
      ),
    ).toBe("hi");
    expect(
      parseOpenAIResponsesSSE(
        JSON.stringify({ type: "response.created", delta: "ignored" }),
      ),
    ).toBe("");
  });

  test("detects reasoning models with provider prefixes", () => {
    expect(isOpenAIReasoningModel("o3-mini")).toBe(true);
    expect(isOpenAIReasoningModel("openai/o4-mini")).toBe(true);
    expect(isOpenAIReasoningModel("openai/gpt-5.5")).toBe(false);
  });

  test("detects Cloudflare Google AI Studio Gemini models", () => {
    expect(
      isCloudflareGoogleAIStudioModel("google-ai-studio/gemini-3.5-flash"),
    ).toBe(true);
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
    expect(isCloudflareAnthropicModel("anthropic/claude-sonnet-4-6")).toBe(
      true,
    );
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

  test("omits deprecated sampling parameters for Cloudflare Anthropic Opus 4.7+ and Sonnet 5", () => {
    expect(
      getOpenAIChatRequestParameters("anthropic/claude-opus-4-8", {
        temperature: 0.5,
        presence_penalty: 0,
        frequency_penalty: 0,
        top_p: 0.8,
      }),
    ).toEqual({});

    expect(
      getOpenAIChatRequestParameters("anthropic/claude-sonnet-5-0", {
        temperature: 0.5,
        presence_penalty: 0,
        frequency_penalty: 0,
        top_p: 0.8,
      }),
    ).toEqual({});
  });
});
