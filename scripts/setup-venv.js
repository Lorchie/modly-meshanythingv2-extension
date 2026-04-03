'use strict'
/**
 * Called by `npm install` postinstall hook.
 * Finds the system Python and runs setup.py to create the extension venv.
 */
const { execFileSync } = require('child_process')
const path = require('path')
const fs   = require('fs')

const EXT_DIR = path.dirname(__dirname)
const IS_WIN  = process.platform === 'win32'

function findPython() {
  const candidates = IS_WIN
    ? ['python', 'python3', 'py']
    : ['python3', 'python']

  for (const cmd of candidates) {
    try {
      execFileSync(cmd, ['--version'], { stdio: 'ignore' })
      return cmd
    } catch { /* try next */ }
  }
  throw new Error(
    'Python 3.10+ is required but was not found on PATH. ' +
    'Install Python and re-run the extension setup.'
  )
}

const venvPy = IS_WIN
  ? path.join(EXT_DIR, 'venv', 'Scripts', 'python.exe')
  : path.join(EXT_DIR, 'venv', 'bin', 'python')

if (fs.existsSync(venvPy)) {
  console.log('[setup-venv] venv already exists, skipping.')
  process.exit(0)
}

const sysPython  = findPython()
const setupScript = path.join(EXT_DIR, 'setup.py')
const args        = JSON.stringify({ python_exe: sysPython, ext_dir: EXT_DIR, gpu_sm: 0 })

console.log(`[setup-venv] Using Python: ${sysPython}`)
console.log('[setup-venv] Running setup.py — this may take several minutes...')

execFileSync(sysPython, [setupScript, args], { stdio: 'inherit', cwd: EXT_DIR })
