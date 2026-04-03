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
    fs.appendFileSync(file, `[${ts}] ${msg}\n`)
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
      const ver = (r.stdout || r.stderr).toString().trim()
      flog(SETUP_LOG, `Found Python: ${cmd} → ${ver}`)
      if (ver.includes('3.10') || ver.includes('3.11') || ver.includes('3.12'))
        return cmd
    }
  }

  throw new Error("Python 3.10–3.12 required but not found")
}

function runSetup(context) {
  flog(SETUP_LOG, "=== runSetup START ===")

  const sysPy = findSystemPython()
  const setupScript = path.join(EXT_DIR, 'setup.py')
  const argsJson = JSON.stringify({ python_exe: sysPy, ext_dir: EXT_DIR })

  return new Promise((resolve, reject) => {
    const child = spawn(sysPy, [setupScript], {
      cwd: EXT_DIR,
      stdio: ['pipe', 'pipe', 'pipe']
    })

    child.stdin.write(argsJson)
    child.stdin.end()

    child.stdout.on('data', d => {
      const msg = d.toString().trim()
      flog(SETUP_LOG, msg)
      context.log(msg)
    })

    child.stderr.on('data', d => {
      const msg = d.toString().trim()
      flog(SETUP_LOG, `[stderr] ${msg}`)
      context.log(msg)
    })

    child.on('close', code => {
      flog(SETUP_LOG, `setup.py exited with ${code}`)
      code === 0 ? resolve() : reject(new Error("setup failed"))
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
    const child = spawn(py, [script], {
      cwd: EXT_DIR,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONPATH: EXT_DIR }
    })

    child.stdin.write(JSON.stringify({
      input,
      params,
      nodeId: context.nodeId,
      workspaceDir: context.workspaceDir,
      tempDir: context.tempDir
    }) + "\n")
    child.stdin.end()

    let buffer = ""

    child.stdout.on('data', chunk => {
      buffer += chunk.toString()
      const lines = buffer.split("\n")
      buffer = lines.pop()

      for (const line of lines) {
        if (!line.trim()) continue
        try {
          const msg = JSON.parse(line)
          if (msg.type === "progress") context.progress(msg.percent, msg.label)
          else if (msg.type === "log") context.log(msg.message)
          else if (msg.type === "done") return resolve(msg.result)
          else if (msg.type === "error") return reject(new Error(msg.message))
        } catch {
          context.log("[raw] " + line)
        }
      }
    })

    child.stderr.on('data', d => context.log("[stderr] " + d.toString()))

    child.on('close', code => {
      if (code !== 0) reject(new Error("Python exited with " + code))
    })
  })
}