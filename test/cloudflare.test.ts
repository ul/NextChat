import { cloudflareAIGatewayUrl } from "../app/utils/cloudflare";

describe("cloudflareAIGatewayUrl", () => {
  test("returns non-gateway urls unchanged", () => {
    const url = "https://api.openai.com/v1/chat/completions";
    expect(cloudflareAIGatewayUrl(url)).toBe(url);
  });

  test("keeps a well-formed openai gateway url intact", () => {
    const url =
      "https://gateway.ai.cloudflare.com/v1/acc/gw/openai/chat/completions";
    expect(cloudflareAIGatewayUrl(url)).toBe(url);
  });

  test("rebuilds an openai gateway url by dropping middle segments", () => {
    const url =
      "https://gateway.ai.cloudflare.com/v1/acc/gw/openai/extra/more/chat/completions";
    expect(cloudflareAIGatewayUrl(url)).toBe(
      "https://gateway.ai.cloudflare.com/v1/acc/gw/openai/chat/completions",
    );
  });

  test("keeps a well-formed azure gateway url intact", () => {
    const url =
      "https://gateway.ai.cloudflare.com/v1/acc/gw/azure-openai/resource/deploy/chat/completions";
    expect(cloudflareAIGatewayUrl(url)).toBe(url);
  });

  test("rebuilds a compat gateway url by dropping the OpenAI v1 segment", () => {
    const url =
      "https://gateway.ai.cloudflare.com/v1/acc/gw/compat/v1/chat/completions";
    expect(cloudflareAIGatewayUrl(url)).toBe(
      "https://gateway.ai.cloudflare.com/v1/acc/gw/compat/chat/completions",
    );
  });
});
