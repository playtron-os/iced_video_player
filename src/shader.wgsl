struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Uniforms {
    rect: vec4<f32>,
    // Pixel size of the (possibly cover-overflowing) draw rect, the visible/clip
    // rect, and the corner radius — for rounded-corner masking of the *visible*
    // area (which, under ContentFit::Cover, is smaller than the draw rect).
    bounds: vec2<f32>,
    clip_size: vec2<f32>,
    radius: f32,
}

@group(0) @binding(0)
var tex_y: texture_2d<f32>;

@group(0) @binding(1)
var tex_uv: texture_2d<f32>;

@group(0) @binding(2)
var s: sampler;

@group(0) @binding(3)
var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var quad = array<vec4<f32>, 6>(
        vec4<f32>(uniforms.rect.xy, 0.0, 0.0),
        vec4<f32>(uniforms.rect.zy, 1.0, 0.0),
        vec4<f32>(uniforms.rect.xw, 0.0, 1.0),
        vec4<f32>(uniforms.rect.zy, 1.0, 0.0),
        vec4<f32>(uniforms.rect.zw, 1.0, 1.0),
        vec4<f32>(uniforms.rect.xw, 0.0, 1.0),
    );

    var out: VertexOutput;
    out.uv = quad[in_vertex_index].zw;
    out.position = vec4<f32>(quad[in_vertex_index].xy, 1.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // BT.709 precomputed coefficents
    let yuv2rgb = mat3x3<f32>(
        1, 0, 1.5748,
        1, -0.1873, -0.4681,
        1, 1.8556, 0,
    );

    var yuv = vec3<f32>(0.0);
    yuv.x = (textureSample(tex_y, s, in.uv).r - 0.0625) / 0.8588;
    yuv.y = (textureSample(tex_uv, s, in.uv).r - 0.5) / 0.8784;
    yuv.z = (textureSample(tex_uv, s, in.uv).g - 0.5) / 0.8784;

    var rgb = clamp(yuv * yuv2rgb, vec3<f32>(0), vec3<f32>(1));

    // Rounded-corner antialiased coverage. Round the *visible* rect — the
    // intersection of the draw rect and the clip rect — centered in the draw
    // rect, so Cover-fit (draw rect overflows the clip) still rounds the visible
    // corners. radius == 0 → coverage is 1 everywhere → opaque (unchanged).
    let px = in.uv * uniforms.bounds;
    let visible = min(uniforms.bounds, uniforms.clip_size);
    let d = rounded_box_sdf(
        2.0 * (px - uniforms.bounds * 0.5),
        visible,
        vec4<f32>(uniforms.radius * 2.0),
    ) / 2.0;
    let alpha = clamp(1.0 - d, 0.0, 1.0);

    // Premultiplied alpha (matches iced's blending convention).
    return vec4<f32>(rgb * alpha, alpha);
}

fn rounded_box_sdf(p: vec2<f32>, size: vec2<f32>, corners: vec4<f32>) -> f32 {
    let box_half = select(corners.yz, corners.xw, p.x > 0.0);
    let corner = select(box_half.y, box_half.x, p.y > 0.0);
    let q = abs(p) - size + corner;
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2(0.0))) - corner;
}
