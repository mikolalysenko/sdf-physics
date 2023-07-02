export function mustHave<T>(x:T|null|undefined) : T {
  if (!x) {
    document.body.innerHTML = `Your browser does not support WebGPU`	
    throw new Error('WebGPU not supported')
  }
  return x
}
mustHave(navigator.gpu)

export function makeCanvas () {
  const canvas = document.createElement('canvas')
  Object.assign(canvas.style, {
    width: '100%',
    height: '100%',
    position: 'absolute',
    left: '0',
    top: '0',
    margin: '0',
    padding: '0'
  })
  canvas.width = window.innerWidth * window.devicePixelRatio
  canvas.height = window.innerHeight * window.devicePixelRatio
  document.body.appendChild(canvas)
  return canvas
}

export type BenchParams = {
  header: string,
  inputs:{[Key:string]:{
    label: string
    props: Partial<HTMLInputElement>
    value?: (v:string, e:HTMLInputElement) => any
  }},
  kernels:{ [Key:string]:any }
}

export type BenchResult<ParamT extends BenchParams> = {
  inputs: {
    [Key in keyof ParamT['inputs']]:(ParamT['inputs'][Key]['value'] extends (v:string) => infer T ? T : string)
  },
  kernel:keyof ParamT['kernels']
}

export function makeBench<ParamT extends BenchParams> ({ header, inputs, kernels }:ParamT) {
  const formContainer = document.createElement('div')
  Object.assign(formContainer.style, {
    padding: '32px',
    margin: '32px'
  })

  const headerNode = document.createElement('div')
  headerNode.innerText = header
  formContainer.appendChild(headerNode)

  const inputElements = Object.entries(inputs).map(([key, {label, props, value}]) => {
    const container = document.createElement('div')
    Object.assign(container.style, {
      padding: '2px'
    })
    const input = document.createElement('input')
    Object.assign(input, props)
    const labelElement = document.createElement('label')
    labelElement.htmlFor = input.id = 'input-' + Math.random()
    labelElement.innerText = label
    container.appendChild(labelElement)
    container.appendChild(input)
    formContainer.appendChild(container)
    return [key, input, value || ((x:string) => x)] as const
  })

  const kernelSelectContainer = document.createElement('div')
  Object.assign(kernelSelectContainer.style, {
    padding: '1px'
  })
  const kernelSelect = document.createElement('select')
  for (const k of Object.keys(kernels)) {
    const opt = document.createElement('option')
    opt.value = opt.text = k
    kernelSelect.appendChild(opt)
  }
  const selectLabel = document.createElement('label')
  selectLabel.innerText = 'Kernel: '
  selectLabel.htmlFor = kernelSelect.id = 'kernel-select'
  kernelSelectContainer.appendChild(selectLabel)
  kernelSelectContainer.appendChild(kernelSelect)
  formContainer.appendChild(kernelSelectContainer)

  const goButton = document.createElement('input')
  goButton.type = 'button'
  goButton.value = 'Go!'
  formContainer.appendChild(goButton)

  const logPre = document.createElement('pre')
  formContainer.appendChild(logPre)

  document.body.appendChild(formContainer)
  
  return {
    sleep (ms:number) {
      return new Promise<void>((resolve) => {
        setTimeout(resolve, ms)
      })
    },

    clear() {
      logPre.innerText = ''
    },

    log (line:string) {
      logPre.innerText += line + '\n'
    },

    go () {
      return new Promise<BenchResult<ParamT>>((resolve) => {
        function handler () {
          const inputs = Object.create(null)
          for (const [key, element, read] of inputElements) {
            inputs[key] = read(element.value, element)
          }
          resolve({
            inputs,
            kernel: kernelSelect.value as any,
          })
        }
        goButton.addEventListener('click', handler)
      })
    }
  }
}