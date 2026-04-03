'use strict'

const { spawn, spawnSync } = require('child_process')
const path = require('path')
const fs   = require('fs')

const EXT_DIR = __dirname
const IS_WIN  = process.platform === 'win32'

const SETUP_LOG   = path.join(EXT_DIR, 'setup.log')
const RUNTIME_LOG = path.join(EXT_DIR, 'runtime.log')

function flog(file, msg) {
  try {
    const ts = new Date().toISOString().slice(11, 19)
    fs.appendFileSync(file, `[${ts}] ${msg}\n`, 'utf8')
  } catch {}
}

function pythonExe() {
  return IS_WIN
    ? path.join(EXT_DIR, 'venv', 'Scripts', 'python.exe')
    : path.join(EXT_DIR, 'venv', 'bin', 'python')
}

function findSystemPython() {
  const candidates = IS_WIN ? ['python', 'py'] : ['python3', 'python']

  for (const cmd of candidates) {
    const r = spawnSync(cmd, ['--version'], { shell: IS_WIN, stdio: 'pipe' })
    if (!r.error && r.status === 0) {
      const ver = (r.stdout || r.stderr || '').toString().trim()
      flog(SETUP_LOG, `Found Python: ${cmd} → ${ver}`)
      // On ne filtre plus par version : on prend le premier qui répond
      return cmd
    }
  }

  throw new Error('No Python interpreter found on PATH')
}

function runSetup(context) {
  flog(SETUP_LOG, '=== runSetup START ===')

  const sysPy       = findSystemPython()
  const setupScript = path.join(EXT_DIR, 'setup.py')
  const argsJson    = JSON.stringify({ python_exe: sysPy, ext_dir: EXT_DIR })

  context.progress(0, 'Preparing Python environment…')
  context.log(`Using system Python: ${sysPy}`)

  return new Promise((resolve, reject) => {
    const child = spawn(sysPy, [setupScript], {
      cwd:   EXT_DIR,
      stdio: ['pipe', 'pipe', 'pipe'],
    })

    child.stdin.write(argsJson)
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

    child.on('close', code => {
      flog(SETUP_LOG, `setup.py exited with code ${code}`)
      if (code === 0) resolve()
      else reject(new Error(`setup.py exited with code ${code}`))
    })
  })
}

module.exports = async function (input, params, context) {
  const py = pythonExe()

  flog(RUNTIME_LOG, `=== run START — nodeId: ${context.nodeId} ===`)
  flog(RUNTIME_LOG, `pythonExe: ${py}`)
  flog(RUNTIME_LOG, `exists: ${fs.existsSync(py)}`)

  if (!fs.existsSync(py)) {
    flog(RUNTIME_LOG, 'venv not found — running setup')
    await runSetup(context)
    flog(RUNTIME_LOG, 'setup completed')
  }

  const script = path.join(EXT_DIR, 'generator.py')
  flog(RUNTIME_LOG, `Spawning generator.py: ${script}`)

  return new Promise((resolve, reject) => {
    const child = spawn(py, [script], {
      cwd:   EXT_DIR,
      stdio: ['pipe', 'pipe', 'pipe'],
      env:   { ...process.env, PYTHONPATH: EXT_DIR },
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
          if (msg.type === 'progress') {
            context.progress(msg.percent, msg.label ?? '')
          } else if (msg.type === 'log') {
            flog(RUNTIME_LOG, msg.message)
            context.log(msg.message)
          } else if (msg.type === 'done') {
            flog(RUNTIME_LOG, `DONE: ${JSON.stringify(msg.result)}`)
            resolved = true
            resolve(msg.result)
          } else if (msg.type === 'error') {
            flog(RUNTIME_LOG, `ERROR: ${msg.message}`)
            resolved = true
            reject(new Error(msg.message))
          }
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

    child.on('close', code => {
      flog(RUNTIME_LOG, `generator.py exited with code ${code}`)
      if (!resolved && code !== 0) {
        reject(new Error(`Python process exited with code ${code}`))
      }
    })
  })
}