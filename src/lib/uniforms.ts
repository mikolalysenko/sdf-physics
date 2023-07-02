export const UNIFORM_SIZES = {
    'mat4x4<f32>': 16,
    'vec4<f32>': 4,
    'vec3<f32>': 3,
    'vec2<f32>': 2,
    'f32': 1
}

export function allocUniforms<Spec extends { [Uniform:string]:keyof typeof UNIFORM_SIZES}, Name extends string>(name:Name, spec:Spec) {
    let count = 0
    for (const type of Object.values(spec)) {
        const size = UNIFORM_SIZES[type]
        if (size) {
            count += size
        }
    }
    const buffer = new Float32Array(count)
    const uniform:{ [Uniform in keyof Spec]:Float32Array } = Object.create(null)
    const wgsl:string[] = []
    let offset = 0
    for (const [key, type] of Object.entries(spec)) {
        const size = UNIFORM_SIZES[type]
        if (!size) {
            continue
        }
        (uniform as any)[key] = buffer.subarray(offset, offset + size)
        offset += size
        wgsl.push(`  ${key} : ${type},`)
    }
    return {
        buffer,
        uniform,
        wgsl: `struct ${name} {
${wgsl.join('\n')}
}`
    }
}