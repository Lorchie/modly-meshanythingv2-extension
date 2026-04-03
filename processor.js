'use strict'

const { spawn } = require('child_process')
const path = require('path')
const fs = require('fs')

const EXT_DIR = __dirname
const IS_WIN  = process.platform === 'win32'

function pythonExe() {
  return IS_WIN
    ? path.join(EXT_DIR, 'venv', 'Scripts', 'python.exe')
    : path.join(EXT_DIR, 'venv', 'bin', 'python')
}

module.exports = async function (input, params, context) {
  const py = pythonExe()

  if (!fs.existsSync(py)) {
    throw new Error(
      'Python environment not found. ' +
      'Uninstall and reinstall the extension from Modly — ' +
      'setup runs automatically during installation.'
    )
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
        extDir:       EXT_DIR,
      }) + '\n'
    )
    child.stdin.end()

    let buffer   = ''
    let resolved = false

    child.stdout.on('data', chunk => {
      buffer += chunk.toString()
      const lines = buffer.split('\n')
      buffer = lines.pop() // keep any incomplete trailing line

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
