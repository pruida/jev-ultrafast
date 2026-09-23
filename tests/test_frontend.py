"""Exercise inspector startup without a browser or paid model calls."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_start_runs_until_done_and_pause_prevents_next_action():
    script = r'''
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
async function check(pause) {
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      value: '', checked: false, handlers: {},
      addEventListener(type, fn) { this.handlers[type] = fn; },
    });
    return elements.get(id);
  }
  const calls = [];
  const context = vm.createContext({
    document: {
      getElementById: element,
      querySelector: () => ({content: 'token'}),
      querySelectorAll: () => [],
    },
    fetch: async () => ({ok: true, json: async () => ({status: 'idle'})}),
    location: {origin: 'http://127.0.0.1:8766'},
    setTimeout,
  });
  vm.runInContext(fs.readFileSync('jev_ultrafast/static/app.js', 'utf8'), context);
  await new Promise(resolve => setImmediate(resolve));
  context.fakeCall = async name => {
    calls.push(name);
    if (pause && name === 'tick') element('stop').handlers.click();
    return {status: name === 'tick' && !pause ? 'done' : 'ready', max_steps: 3};
  };
  vm.runInContext(`
    render = () => {};
    call = async (name, body) => { state = await fakeCall(name, body); return state; };
  `, context);
  element('task-form').handlers.submit({preventDefault() {}});
  for (let i = 0; i < 10; i++) await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(calls, pause ? ['reset', 'tick'] : ['reset', 'tick', 'report']);
}
(async () => { await check(false); await check(true); })().catch(error => {
  console.error(error); process.exitCode = 1;
});
'''
    subprocess.run(["node", "-e", script], cwd=ROOT, check=True, capture_output=True, text=True)
