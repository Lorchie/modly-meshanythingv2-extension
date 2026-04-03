'use strict'

const { spawn, spawnSync } = require('child_process')
const path = require('path')
const fs   = require('fs')

const EXT_DIR = __dirname
const IS_WIN  = process.platform === 'win32'

function pythonExe() {
  return IS_WIN
    ? path.join(EXT_DIR, 'venv', 'Scripts', 'python.exe')
    : path.join(EXT_DIR, 'venv', 'bin', 'python')
}

function findSystemPython() {
  const candidates = IS_WIN
    ? ['python', 'py', 'python3']
    : ['python3', 'python']

  for (const cmd of candidates) {
    const r = spawnSync(cmd, ['--version'], { shell: IS_WIN, stdio: 'pipe' })
    if (!r.error && r.status === 0) return cmd
  }

  throw new Error(
    'Python 3.10+ is required but was not found on PATH. ' +
    'Please install Python and try again.'
  )
}

function runSetup(context) {
  context.progress(0, 'First run: setting up Python environment (this may take several minutes)...')
  context.log('Creating venv and installing dependencies...')

  const sysPy      = findSystemPython()
  const setupScript = path.join(EXT_DIR, 'setup.py')
  const args        = JSON.stringify({ python_exe: sysPy, ext_dir: EXT_DIR, gpu_sm: 0 })

  context.log(`Using Python: ${sysPy}`)

  return new Promise((resolve, reject) => {
    const child = spawn(sysPy, [setupScript], {
      cwd:   EXT_DIR,
      stdio: ['pipe', 'pipe', 'pipe'],
    })

    child.stdin.write(args)
    child.stdin.end()

    child.stdout.on('data', chunk => context.log(chunk.toString().trimEnd()))
    child.stderr.on('data', chunk => context.log(chunk.toString().trimEnd()))

    child.on('error', err => reject(new Error(`setup failed: ${err.message}`)))
    child.on('close', code => {
      if (code === 0) resolve()
      else reject(new Error(`setup.py exited with code ${code}`))
    })
  })
}

module.exports = async function (input, params, context) {
  const py = pythonExe()

  if (!fs.existsSync(py)) {
    await runSetup(context)
  }

  const script = path.join(EXT_DIR, 'generator.py')

  return new Promise((resolve, reject) => {
    const child = spawn(py, [script], { cwd: EXT_DIR })

    child.stdin.write(
      JSON.stringify({
        input,
        params,
        nodeId:       context.nodeId,
        workspaceDir: context.workspaceDir,
        tempDir:      context.tempDir,
      }) + '\n'
    )
    child.stdin.end()

    let buffer   = ''
    let resolved = false

    child.stdout.on('data', chunk => {
      buffer += chunk.toString()
      const lines = buffer.split('\n')
      buffer = lines.pop()

      for (const line of lines) {
        if (!line.trim()) continue
        try {
          const msg = JSON.parse(line)
          if      (msg.type === 'progress') context.progress(msg.percent, msg.label ?? '')
          else if (msg.type === 'log')      context.log(msg.message)
          else if (msg.type === 'done')   { resolved = true; resolve(msg.result) }
          else if (msg.type === 'error')  { resolved = true; reject(new Error(msg.message)) }
        } catch {
          context.log(`[raw] ${line}`)
        }
      }
    })

    child.stderr.on('data', chunk => {
      context.log(`[stderr] ${chunk.toString().trimEnd()}`)
    })

    child.on('error', err => {
      if (!resolved) { resolved = true; reject(err) }
    })

    child.on('close', code => {
      if (!resolved && code !== 0) {
        reject(new Error(`Python process exited with code ${code}`))
      }
    })
  })
}
