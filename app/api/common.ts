import { NextRequest, NextResponse } from "next/server";
import { getServerSideConfig } from "../config/server";
import { OPENAI_BASE_URL, ServiceProvider } from "../constant";
import { cloudflareAIGatewayUrl } from "../utils/cloudflare";
import { getModelProvider, isModelNotavailableInServer } from "../utils/model";
import {
  isCloudflareAnthropicModel,
  isCloudflareGoogleAIStudioModel,
} from "../utils/openai";

const serverConfig = getServerSideConfig();

function asBearerToken(token: string) {
  return token.startsWith("Bearer ") ? token : `Bearer ${token}`;
}

function getCloudflareCompatProviderApiKey(model?: string) {
  if (!model) return "";

  if (isCloudflareGoogleAIStudioModel(model)) {
    return serverConfig.googleApiKey || "";
  }

  if (isCloudflareAnthropicModel(model)) {
    return serverConfig.anthropicApiKey || "";
  }

  if (model.trim().toLowerCase().startsWith("openai/")) {
    return serverConfig.apiKey || "";
  }

  return "";
}

type OpenAIChatMessage = {
  role?: string;
  content?: unknown;
};

type OpenAIChatBody = {
  model?: string;
  messages?: OpenAIChatMessage[];
  temperature?: number;
  max_tokens?: number;
  max_completion_tokens?: number;
} & Record<string, unknown>;

type GeminiPart =
  | { text: string }
  | { inlineData: { mimeType: string; data: string } };

type GeminiContent = {
  role: "user" | "model";
  parts: GeminiPart[];
};

function parseCloudflareCompatGateway(fetchUrl: string) {
  try {
    const url = new URL(fetchUrl);
    if (url.hostname !== "gateway.ai.cloudflare.com") return {};

    const parts = url.pathname.split("/").filter(Boolean);
    const compatIndex = parts.indexOf("compat");
    if (parts[0] !== "v1" || compatIndex < 3) return {};

    return {
      accountId: parts[1],
      gatewayId: parts[2],
    };
  } catch (e) {
    return {};
  }
}

function getCloudflareAIRunModel(model?: string) {
  return model?.replace(/^google-ai-studio\//i, "google/") ?? "";
}

function dataUrlToGeminiPart(url: string): GeminiPart | undefined {
  const match = url.match(/^data:([^;,]+);base64,(.+)$/);
  if (!match) return undefined;

  return {
    inlineData: {
      mimeType: match[1],
      data: match[2],
    },
  };
}

function openAIContentToGeminiParts(content: unknown): GeminiPart[] {
  if (typeof content === "string") {
    return content ? [{ text: content }] : [];
  }

  if (!Array.isArray(content)) return [];

  const parts: GeminiPart[] = [];
  for (const part of content) {
    if (typeof part === "string") {
      if (part) parts.push({ text: part });
      continue;
    }

    if (!part || typeof part !== "object") continue;

    const typedPart = part as {
      type?: string;
      text?: string;
      image_url?: { url?: string };
    };

    if (typedPart.type === "text" && typedPart.text) {
      parts.push({ text: typedPart.text });
      continue;
    }

    const imageUrl = typedPart.image_url?.url;
    if (typedPart.type === "image_url" && imageUrl) {
      const imagePart = dataUrlToGeminiPart(imageUrl);
      if (imagePart) parts.push(imagePart);
    }
  }

  return parts;
}

function openAIContentToText(content: unknown) {
  return openAIContentToGeminiParts(content)
    .map((part) => ("text" in part ? part.text : ""))
    .filter(Boolean)
    .join("\n\n");
}

function buildCloudflareGoogleAIRunBody(body: OpenAIChatBody) {
  const systemTexts: string[] = [];
  const contents: GeminiContent[] = [];

  for (const message of body.messages ?? []) {
    const role = (message.role || "user").toLowerCase();

    if (role === "system" || role === "developer") {
      const text = openAIContentToText(message.content);
      if (text) systemTexts.push(text);
      continue;
    }

    const parts = openAIContentToGeminiParts(message.content);
    if (!parts.length) continue;

    const geminiRole = role === "assistant" ? "model" : "user";
    const previous = contents.at(-1);
    if (previous?.role === geminiRole) {
      previous.parts.push(...parts);
    } else {
      contents.push({ role: geminiRole, parts });
    }
  }

  const generationConfig: Record<string, unknown> = {};
  if (typeof body.temperature === "number") {
    generationConfig.temperature = body.temperature;
  }

  const maxOutputTokens = body.max_completion_tokens ?? body.max_tokens;
  if (typeof maxOutputTokens === "number") {
    generationConfig.maxOutputTokens = maxOutputTokens;
  }

  return {
    model: getCloudflareAIRunModel(body.model),
    input: {
      ...(systemTexts.length
        ? { systemInstruction: { parts: [{ text: systemTexts.join("\n\n") }] } }
        : {}),
      contents,
      ...(Object.keys(generationConfig).length ? { generationConfig } : {}),
    },
  };
}

function extractCloudflareGoogleText(body: any) {
  const payload = body?.result ?? body;

  const openAIText = payload?.choices?.at?.(0)?.message?.content;
  if (typeof openAIText === "string") return openAIText;

  const candidates = payload?.candidates;
  if (!Array.isArray(candidates)) return "";

  return (
    candidates
      .at(0)
      ?.content?.parts?.map((part: any) => part?.text || "")
      .filter(Boolean)
      .join("\n\n") ?? ""
  );
}

async function normalizeCloudflareAIRunResponse(res: Response) {
  const responseText = await res.text();
  const headers = new Headers(res.headers);
  headers.delete("www-authenticate");
  headers.delete("content-encoding");
  headers.delete("content-length");
  headers.set("X-Accel-Buffering", "no");
  headers.set("Content-Type", "application/json");

  if (!res.ok) {
    return new Response(responseText, {
      status: res.status,
      statusText: res.statusText,
      headers,
    });
  }

  try {
    const json = JSON.parse(responseText);
    const responseErrors = Array.isArray(json?.errors)
      ? json.errors
      : json?.errors
      ? [json.errors]
      : [];
    if (json?.error || responseErrors.length > 0 || json?.success === false) {
      return new Response(
        JSON.stringify({
          error: json?.error ?? (responseErrors.length ? responseErrors : json),
        }),
        {
          status: res.status,
          statusText: res.statusText,
          headers,
        },
      );
    }

    const content = extractCloudflareGoogleText(json) || responseText;

    return new Response(
      JSON.stringify({
        choices: [
          {
            message: {
              content,
            },
          },
        ],
      }),
      {
        status: res.status,
        statusText: res.statusText,
        headers,
      },
    );
  } catch (e) {
    return new Response(responseText, {
      status: res.status,
      statusText: res.statusText,
      headers,
    });
  }
}

export async function requestOpenai(req: NextRequest) {
  const controller = new AbortController();

  const isAzure = req.nextUrl.pathname.includes("azure/deployments");

  var authValue,
    authHeaderName = "";
  if (isAzure) {
    authValue =
      req.headers
        .get("Authorization")
        ?.trim()
        .replaceAll("Bearer ", "")
        .trim() ?? "";

    authHeaderName = "api-key";
  } else {
    authValue = req.headers.get("Authorization") ?? "";
    authHeaderName = "Authorization";
  }

  let path = `${req.nextUrl.pathname}`.replaceAll("/api/openai/", "");

  let baseUrl =
    (isAzure ? serverConfig.azureUrl : serverConfig.baseUrl) || OPENAI_BASE_URL;

  if (!baseUrl.startsWith("http")) {
    baseUrl = `https://${baseUrl}`;
  }

  if (baseUrl.endsWith("/")) {
    baseUrl = baseUrl.slice(0, -1);
  }

  console.log("[Proxy] ", path);
  console.log("[Base Url]", baseUrl);

  const timeoutId = setTimeout(
    () => {
      controller.abort();
    },
    10 * 60 * 1000,
  );

  if (isAzure) {
    const azureApiVersion =
      req?.nextUrl?.searchParams?.get("api-version") ||
      serverConfig.azureApiVersion;
    baseUrl = baseUrl.split("/deployments").shift() as string;
    path = `${req.nextUrl.pathname.replaceAll(
      "/api/azure/",
      "",
    )}?api-version=${azureApiVersion}`;

    // Forward compatibility:
    // if display_name(deployment_name) not set, and '{deploy-id}' in AZURE_URL
    // then using default '{deploy-id}'
    if (serverConfig.customModels && serverConfig.azureUrl) {
      const modelName = path.split("/")[1];
      let realDeployName = "";
      serverConfig.customModels
        .split(",")
        .filter(
          (v: string) => !!v && !v.startsWith("-") && v.includes(modelName),
        )
        .forEach((m: string) => {
          const [fullName, displayName] = m.split("=");
          const [_, providerName] = getModelProvider(fullName);
          if (providerName === "azure" && !displayName) {
            const [_, deployId] = (serverConfig?.azureUrl ?? "").split(
              "deployments/",
            );
            if (deployId) {
              realDeployName = deployId;
            }
          }
        });
      if (realDeployName) {
        console.log("[Replace with DeployId", realDeployName);
        path = path.replaceAll(modelName, realDeployName);
      }
    }
  }

  const fetchUrl = cloudflareAIGatewayUrl(`${baseUrl}/${path}`);
  const isCloudflareAIRestApi =
    !isAzure &&
    fetchUrl.startsWith("https://api.cloudflare.com/client/v4/accounts/") &&
    fetchUrl.includes("/ai/v1/");
  const isCloudflareAICompatApi =
    !isAzure &&
    fetchUrl.startsWith("https://gateway.ai.cloudflare.com/") &&
    fetchUrl.includes("/compat/");

  let requestBody: BodyInit | null | undefined = req.body;
  let jsonBody: ({ model?: string } & Record<string, unknown>) | undefined;
  if (req.body) {
    try {
      const clonedBody = await req.text();
      requestBody = clonedBody;
      jsonBody = JSON.parse(clonedBody) as { model?: string } & Record<
        string,
        unknown
      >;
    } catch (e) {
      console.error("[OpenAI] request body parse", e);
    }
  }

  const incomingCloudflareAIGatewayAuthValue =
    req.headers.get("cf-aig-authorization") || "";
  const cloudflareAIGatewayAuthValue = serverConfig.cloudflareAIGatewayApiKey
    ? asBearerToken(serverConfig.cloudflareAIGatewayApiKey)
    : incomingCloudflareAIGatewayAuthValue;
  const providerAuthValue =
    isCloudflareAICompatApi && jsonBody?.model
      ? getCloudflareCompatProviderApiKey(jsonBody.model)
      : "";
  const incomingProviderAuthValue = authValue;

  const requestAuthValue = isCloudflareAIRestApi
    ? cloudflareAIGatewayAuthValue || authValue
    : isCloudflareAICompatApi
    ? providerAuthValue
      ? asBearerToken(providerAuthValue)
      : cloudflareAIGatewayAuthValue || incomingProviderAuthValue
    : authValue;

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "Cache-Control": "no-store",
    ...(requestAuthValue && { [authHeaderName]: requestAuthValue }),
    ...(isCloudflareAICompatApi &&
      cloudflareAIGatewayAuthValue &&
      requestAuthValue !== cloudflareAIGatewayAuthValue && {
        "cf-aig-authorization": cloudflareAIGatewayAuthValue,
      }),
    ...(serverConfig.openaiOrgId && {
      "OpenAI-Organization": serverConfig.openaiOrgId,
    }),
  };

  console.log("fetchUrl", fetchUrl);
  const fetchOptions: RequestInit = {
    headers,
    method: req.method,
    body: requestBody,
    // to fix #2485: https://stackoverflow.com/questions/55920957/cloudflare-worker-typeerror-one-time-use-body
    redirect: "manual",
    // @ts-ignore
    duplex: "half",
    signal: controller.signal,
  };

  // #1815 try to refuse gpt4 request
  if (serverConfig.customModels && jsonBody) {
    try {
      // not undefined and is false
      if (
        isModelNotavailableInServer(
          serverConfig.customModels,
          jsonBody?.model as string,
          [
            ServiceProvider.OpenAI,
            ServiceProvider.Azure,
            jsonBody?.model as string, // support provider-unspecified model
          ],
        )
      ) {
        return NextResponse.json(
          {
            error: true,
            message: `you are not allowed to use ${jsonBody?.model} model`,
          },
          {
            status: 403,
          },
        );
      }
    } catch (e) {
      console.error("[OpenAI] gpt4 filter", e);
    }
  }

  const shouldUseCloudflareAIRun =
    isCloudflareAICompatApi &&
    isCloudflareGoogleAIStudioModel(jsonBody?.model ?? "");

  try {
    if (shouldUseCloudflareAIRun && jsonBody) {
      const gatewayConfig = parseCloudflareCompatGateway(fetchUrl);
      const accountId =
        serverConfig.cloudflareAccountId || gatewayConfig.accountId;
      const gatewayId =
        serverConfig.cloudflareAIGatewayId || gatewayConfig.gatewayId;
      const restAuthValue = cloudflareAIGatewayAuthValue || authValue;

      if (accountId && gatewayId && restAuthValue) {
        const res = await fetch(
          `https://api.cloudflare.com/client/v4/accounts/${accountId}/ai/run`,
          {
            headers: {
              "Content-Type": "application/json",
              "Cache-Control": "no-store",
              Authorization: restAuthValue,
              "cf-aig-gateway-id": gatewayId,
            },
            method: "POST",
            body: JSON.stringify(
              buildCloudflareGoogleAIRunBody(jsonBody as OpenAIChatBody),
            ),
            redirect: "manual",
            signal: controller.signal,
          },
        );

        return await normalizeCloudflareAIRunResponse(res);
      }

      console.warn(
        "[Cloudflare AI Run] missing account id, gateway id, or Cloudflare token; falling back to compat endpoint",
      );
    }

    const res = await fetch(fetchUrl, fetchOptions);

    // Extract the OpenAI-Organization header from the response
    const openaiOrganizationHeader = res.headers.get("OpenAI-Organization");

    // Check if serverConfig.openaiOrgId is defined and not an empty string
    if (serverConfig.openaiOrgId && serverConfig.openaiOrgId.trim() !== "") {
      // If openaiOrganizationHeader is present, log it; otherwise, log that the header is not present
      console.log("[Org ID]", openaiOrganizationHeader);
    } else {
      console.log("[Org ID] is not set up.");
    }

    // to prevent browser prompt for credentials
    const newHeaders = new Headers(res.headers);
    newHeaders.delete("www-authenticate");
    // to disable nginx buffering
    newHeaders.set("X-Accel-Buffering", "no");

    // Conditionally delete the OpenAI-Organization header from the response if [Org ID] is undefined or empty (not setup in ENV)
    // Also, this is to prevent the header from being sent to the client
    if (!serverConfig.openaiOrgId || serverConfig.openaiOrgId.trim() === "") {
      newHeaders.delete("OpenAI-Organization");
    }

    // The latest version of the OpenAI API forced the content-encoding to be "br" in json response
    // So if the streaming is disabled, we need to remove the content-encoding header
    // Because Vercel uses gzip to compress the response, if we don't remove the content-encoding header
    // The browser will try to decode the response with brotli and fail
    newHeaders.delete("content-encoding");

    return new Response(res.body, {
      status: res.status,
      statusText: res.statusText,
      headers: newHeaders,
    });
  } finally {
    clearTimeout(timeoutId);
  }
}
