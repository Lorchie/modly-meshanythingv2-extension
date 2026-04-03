'use strict'

const { spawn, spawnSync } = require('child_process')
const path = require('path')
const fs   = require('fs')

const EXT_DIR = __dirname
const IS_WIN  = process.platform === 'win32'

// ── File logger — works even before generator.py starts ──────────────────────

const SETUP_LOG   = path.join(EXT_DIR, 'setup.log')
const RUNTIME_LOG = path.join(EXT_DIR, 'runtime.log')

function flog(file, message) {
  try {
    const ts   = new Date().toISOString().slice(11, 19)
    fs.appendFileSync(file, `[${ts}] ${message}\n`, 'utf8')
  } catch {}
}

// ─────────────────────────────────────────────────────────────────────────────

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
    if (!r.error && r.status === 0) {
      const ver = (r.stdout || r.stderr || '').toString().trim()
      flog(SETUP_LOG, `Found Python: ${cmd} → ${ver}`)
      return cmd
    }
  }

  throw new Error(
    'Python 3.10+ is required but was not found on PATH. ' +
    'Please install Python and try again.'
  )
}

function runSetup(context) {
  flog(SETUP_LOG, '=== runSetup START ===')
  flog(SETUP_LOG, `EXT_DIR: ${EXT_DIR}`)
  flog(SETUP_LOG, `pythonExe target: ${pythonExe()}`)

  context.progress(0, 'First run: setting up Python environment (this may take several minutes)...')
  context.log('Creating venv and installing dependencies...')

  let sysPy
  try {
    sysPy = findSystemPython()
  } catch (err) {
    flog(SETUP_LOG, `ERROR findSystemPython: ${err.message}`)
    throw err
  }

  const setupScript = path.join(EXT_DIR, 'setup.py')
  const args        = JSON.stringify({ python_exe: sysPy, ext_dir: EXT_DIR, gpu_sm: 0 })

  flog(SETUP_LOG, `Using Python: ${sysPy}`)
  flog(SETUP_LOG, `setup.py: ${setupScript}`)
  flog(SETUP_LOG, `args: ${args}`)
  context.log(`Using Python: ${sysPy}`)

  return new Promise((resolve, reject) => {
    flog(SETUP_LOG, 'Spawning setup.py...')

    const child = spawn(sysPy, [setupScript], {
      cwd:   EXT_DIR,
      stdio: ['pipe', 'pipe', 'pipe'],
    })

    child.stdin.write(args)
    child.stdin.end()

    child.stdout.on('data', chunk => {
      const msg = chunk.toString().trimEnd()
      flog(SETUP_LOG, msg)
      context.log(msg)
    })

    child.stderr.on('data', chunk => {
      const msg = chunk.toString().trimEnd()
      flog(SETUP_LOG, `[stderr] ${msg}`)
      context.log(msg)
    })

    child.on('error', err => {
      flog(SETUP_LOG, `spawn error: ${err.message}`)
      reject(new Error(`setup failed: ${err.message}`))
    })

    child.on('close', code => {
      flog(SETUP_LOG, `setup.py exited with code ${code}`)
      if (code === 0) {
        flog(SETUP_LOG, '=== runSetup DONE ===')
        resolve()
      } else {
        reject(new Error(`setup.py exited with code ${code}`))
      }
    })
  })
}

module.exports = async function (input, params, context) {
  const py = pythonExe()

  flog(RUNTIME_LOG, `=== run START — nodeId: ${context.nodeId} ===`)
  flog(RUNTIME_LOG, `pythonExe: ${py}`)
  flog(RUNTIME_LOG, `exists: ${fs.existsSync(py)}`)

  if (!fs.existsSync(py)) {
    flog(RUNTIME_LOG, 'venv not found — launching runSetup()')
    await runSetup(context)
    flog(RUNTIME_LOG, 'runSetup() completed')
  }

  const script = path.join(EXT_DIR, 'generator.py')
  flog(RUNTIME_LOG, `Spawning generator.py: ${script}`)

  return new Promise((resolve, reject) => {
    const child = spawn(py, [script], {
      cwd:   EXT_DIR,
      stdio: ['pipe', 'pipe', 'pipe'],
    })

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
          else if (msg.type === 'log')      { flog(RUNTIME_LOG, msg.message); context.log(msg.message) }
          else if (msg.type === 'done')   { flog(RUNTIME_LOG, `DONE: ${JSON.stringify(msg.result)}`); resolved = true; resolve(msg.result) }
          else if (msg.type === 'error')  { flog(RUNTIME_LOG, `ERROR: ${msg.message}`); resolved = true; reject(new Error(msg.message)) }
        } catch {
          flog(RUNTIME_LOG, `[raw] ${line}`)
          context.log(`[raw] ${line}`)
        }
      }
    })

    child.stderr.on('data', chunk => {
      const msg = chunk.toString().trimEnd()
      flog(RUNTIME_LOG, `[stderr] ${msg}`)
      context.log(`[stderr] ${msg}`)
    })

    child.on('error', err => {
      flog(RUNTIME_LOG, `spawn error: ${err.message}`)
      if (!resolved) { resolved = true; reject(err) }
    })

    child.on('close', code => {
      flog(RUNTIME_LOG, `generator.py exited with code ${code}`)
      if (!resolved && code !== 0) {
        reject(new Error(`Python process exited with code ${code}`))
      }
    })
  })
}
