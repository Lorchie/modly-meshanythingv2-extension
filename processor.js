const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

module.exports = async function ({ node, inputs, params, context }) {
  if (!context) {
    throw new Error("Context object is missing. Modly did not pass context.");
  }

  const extDir = context.extensionDir;
  if (!extDir) {
    throw new Error("context.extensionDir is undefined.");
  }

  const venvPython =
    process.platform === "win32"
      ? path.join(extDir, "venv", "Scripts", "python.exe")
      : path.join(extDir, "venv", "bin", "python");

  const generatorPath = path.join(extDir, "generator.py");

  const inputMesh = inputs.mesh;
  if (!inputMesh) throw new Error("No input mesh provided.");

  const outPath = path.join(
    context.workDir,
    `meshanythingv2_${node.id}_output.obj`
  );

  const args = [
    generatorPath,
    node.id,
    inputMesh,
    outPath,
    JSON.stringify(params || {})
  ];

  await new Promise((resolve, reject) => {
    const proc = spawn(venvPython, args, {
      cwd: extDir,
      stdio: ["ignore", "pipe", "pipe"]
    });

    proc.stdout.on("data", (data) => {
      context.log(`[meshanythingv2][stdout] ${data}`);
    });

    proc.stderr.on("data", (data) => {
      context.log(`[meshanythingv2][stderr] ${data}`);
    });

    proc.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`generator.py exited with code ${code}`));
    });
  });

  if (!fs.existsSync(outPath)) {
    throw new Error("MeshAnythingV2 did not produce an output mesh.");
  }

  return { mesh: outPath };
};