import { mat4, vec3 } from 'gl-matrix'
import { allocUniforms } from './uniforms'

export class RenderUniforms {
    public uniforms = allocUniforms('render', {
        view : 'mat4x4<f32>',
        proj : 'mat4x4<f32>',
        projInv: 'mat4x4<f32>',
        fog : 'vec4<f32>',
        lightDir : 'vec4<f32>',
        eye : 'vec4<f32>'
    })
    
    constructor () {
    }

    public updateBindGroup () {
    }

    public uniformStruct () {
    }
}

export class RenderSDF {

}