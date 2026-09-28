struct VSInput
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
    [[vk::location(2)]] float3 color : COLOR;
};

struct VSOutput
{
    float4 position : SV_Position;
    float3 worldNormal : NORMAL;
    float3 color : COLOR;
};

struct CubeUBO
{
    float4x4 mvp;
    float4x4 model;
    float4 lightDir;
};

[[vk::binding(0, 0)]] ConstantBuffer<CubeUBO> ubo;

[shader("vertex")]
VSOutput vertMain(VSInput input)
{
    VSOutput output;
    output.position = mul(ubo.mvp, float4(input.position, 1.0));
    output.worldNormal = mul((float3x3)ubo.model, input.normal);
    output.color = input.color;
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target
{
    float3 normal = normalize(input.worldNormal);
    float3 light = normalize(ubo.lightDir.xyz);
    float wrap = saturate(dot(normal, light)) * 0.72 + 0.28;
    float3 color = input.color * wrap;
    return float4(color, 1.0);
}
