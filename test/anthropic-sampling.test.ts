import {
  getAnthropicSamplingParameters,
  isAnthropicSamplingUnsupportedModel,
} from "../app/utils/anthropic";

describe("getAnthropicSamplingParameters", () => {
  test("omits default top_p when temperature is set", () => {
    expect(
      getAnthropicSamplingParameters({ temperature: 0.5, top_p: 1 }),
    ).toEqual({ temperature: 0.5 });
  });

  test("prefers explicit top_p over temperature", () => {
    expect(
      getAnthropicSamplingParameters({ temperature: 0.5, top_p: 0.8 }),
    ).toEqual({ top_p: 0.8 });
  });

  test("keeps top_p when temperature is absent", () => {
    expect(getAnthropicSamplingParameters({ top_p: 1 })).toEqual({ top_p: 1 });
  });

  test("detects Claude models that do not support sampling parameters", () => {
    expect(
      isAnthropicSamplingUnsupportedModel("anthropic/claude-opus-4-8"),
    ).toBe(true);
    expect(isAnthropicSamplingUnsupportedModel("claude-opus-4.7")).toBe(
      true,
    );
    expect(
      isAnthropicSamplingUnsupportedModel("anthropic/claude-sonnet-5-0"),
    ).toBe(true);
    expect(isAnthropicSamplingUnsupportedModel("claude-sonnet-5.0")).toBe(
      true,
    );
    expect(
      isAnthropicSamplingUnsupportedModel("anthropic/claude-sonnet-4-6"),
    ).toBe(false);
  });

  test("omits sampling parameters for Claude models that reject them", () => {
    expect(
      getAnthropicSamplingParameters(
        { temperature: 0.5, top_p: 0.8 },
        "anthropic/claude-opus-4-8",
      ),
    ).toEqual({});

    expect(
      getAnthropicSamplingParameters(
        { temperature: 0.5, top_p: 0.8 },
        "anthropic/claude-sonnet-5-0",
      ),
    ).toEqual({});
  });
});
