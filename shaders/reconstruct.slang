struct VSOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

struct ReconstructUBO
{
    float4x4 invViewProj;
    float4x4 prevViewProj;
    float4x4 objectReproject;
    float2 renderSize;
    float2 outputSize;
    uint frameIndex;
    uint firstFrame;
    uint mode;
    uint _pad;
};

[[vk::binding(0, 0)]] Texture2D<float4> currentColor;
[[vk::binding(1, 0)]] Texture2D<float> currentDepth;
[[vk::binding(2, 0)]] Texture2D<float4> prevColor;
[[vk::binding(3, 0)]] Texture2D<float> prevDepth;
[[vk::binding(4, 0)]] SamplerState linearClamp;
[[vk::binding(5, 0)]] SamplerState nearestClamp;
[[vk::binding(6, 0)]] ConstantBuffer<ReconstructUBO> ubo;

[shader("vertex")]
VSOutput vertMain(uint vertexId : SV_VertexID)
{
    VSOutput output;
    float2 uv = float2((vertexId << 1) & 2, vertexId & 2);
    output.uv = uv;
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

float3 neighborhoodAverage(float2 uv, float2 texel)
{
    float3 left = currentColor.SampleLevel(linearClamp, uv + float2(-texel.x, 0.0), 0).rgb;
    float3 right = currentColor.SampleLevel(linearClamp, uv + float2(texel.x, 0.0), 0).rgb;
    float3 up = currentColor.SampleLevel(linearClamp, uv + float2(0.0, -texel.y), 0).rgb;
    float3 down = currentColor.SampleLevel(linearClamp, uv + float2(0.0, texel.y), 0).rgb;
    return (left + right + up + down) * 0.25;
}

void neighborhoodMinMax(float2 uv, float2 texel, float3 center, out float3 nmin, out float3 nmax)
{
    float3 left = currentColor.SampleLevel(linearClamp, uv + float2(-texel.x, 0.0), 0).rgb;
    float3 right = currentColor.SampleLevel(linearClamp, uv + float2(texel.x, 0.0), 0).rgb;
    float3 up = currentColor.SampleLevel(linearClamp, uv + float2(0.0, -texel.y), 0).rgb;
    float3 down = currentColor.SampleLevel(linearClamp, uv + float2(0.0, texel.y), 0).rgb;
    nmin = min(center, min(min(left, right), min(up, down)));
    nmax = max(center, max(max(left, right), max(up, down)));
}

float2 unprojectUvToNdc(float2 uv)
{
    return float2(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0);
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target
{
    float2 uv = input.uv;
    float3 current = currentColor.SampleLevel(linearClamp, uv, 0).rgb;

    // mode 0: naive bilinear upsample of the 480p target
    if (ubo.mode == 0)
    {
        return float4(current, 1.0);
    }

    int2 pixel = int2(uv * ubo.outputSize);
    float2 texel = 1.0 / ubo.renderSize;
    uint checker = (uint(pixel.x) + uint(pixel.y) + ubo.frameIndex) & 1;

    // Fresh checkerboard cells keep the current 480p sample.
    if (ubo.firstFrame != 0 || checker == 0)
    {
        return float4(current, 1.0);
    }

    int2 low = int2(clamp(uv * ubo.renderSize, float2(0.0, 0.0), ubo.renderSize - 1.0));
    float depth = currentDepth.Load(int3(low, 0));

    float2 ndc = unprojectUvToNdc(uv);
    float4 worldH = mul(ubo.invViewProj, float4(ndc, depth, 1.0));
    float3 world = worldH.xyz / max(worldH.w, 1e-6);

    float4 prevClip;
    if (depth >= 0.999)
    {
        prevClip = mul(ubo.prevViewProj, float4(world, 1.0));
    }
    else
    {
        prevClip = mul(ubo.objectReproject, float4(world, 1.0));
    }

    float invW = 1.0 / max(prevClip.w, 1e-6);
    float2 prevNdc = prevClip.xy * invW;
    float2 prevUv = float2(prevNdc.x * 0.5 + 0.5, prevNdc.y * 0.5 + 0.5);
    float prevZ = prevClip.z * invW;

    bool inside = all(prevUv >= 0.0) && all(prevUv <= 1.0);
    if (!inside)
    {
        return float4(neighborhoodAverage(uv, texel), 1.0);
    }

    float historyDepth = prevDepth.SampleLevel(linearClamp, prevUv, 0);
    if (abs(prevZ - historyDepth) > 0.012)
    {
        return float4(neighborhoodAverage(uv, texel), 1.0);
    }

    float3 history = prevColor.SampleLevel(linearClamp, prevUv, 0).rgb;
    float3 nmin, nmax;
    neighborhoodMinMax(uv, texel, current, nmin, nmax);
    history = clamp(history, nmin, nmax);

    // Blend slightly toward the current sample so edges stay stable.
    float3 reconstructed = lerp(history, current, 0.18);
    return float4(reconstructed, 1.0);
}
