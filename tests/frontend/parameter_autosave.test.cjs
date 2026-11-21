const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../../web/app.js'), 'utf8');
const helpers = source.slice(source.indexOf('const parameterSaves = new Map();'),
                             source.indexOf('async function act('));

function environment() {
  const timers = new Map(), calls = [], errors = [];
  let identifier = 0;
  const context = vm.createContext({
    Map, generation: 0,
    setTimeout: fn => { timers.set(++identifier, fn); return identifier; },
    clearTimeout: id => timers.delete(id),
    api: (url, data) => new Promise((resolve, reject) => {
      calls.push({url, data: JSON.parse(JSON.stringify(data)), resolve, reject});
    }),
    refreshStatus: async () => {},
    notice: message => errors.push(message),
  });
  vm.runInContext(helpers, context);
  return {
    calls, errors,
    bind(key = 'task/example') {
      const handlers = new Map();
      const values = {amount: 1, valid: true, dirty: false, saved: 0};
      const form = {
        isConnected: true, checkValidity: () => values.valid,
        addEventListener: (name, fn) => handlers.set(name, fn),
      };
      context.autoSaveForm(form, key, '/api/features/change',
        () => ({amount: values.amount}), () => { values.dirty = true; },
        () => { values.dirty = false; values.saved++; });
      return {
        form, values,
        input(type = 'number') {
          handlers.get('input')({target: {type, tagName: 'INPUT'}});
        },
        change() { handlers.get('change')(); },
      };
    },
    tick() {
      const due = [...timers.values()];
      timers.clear();
      due.forEach(fn => fn());
    },
  };
}

async function settle() {
  await new Promise(resolve => setImmediate(resolve));
}

test('typing saves the latest value once and matching change events are deduplicated', async () => {
  const env = environment(), field = env.bind();
  for (const value of [2, 20, 200]) {
    field.values.amount = value;
    field.input();
  }
  assert.equal(env.calls.length, 0);
  env.tick();
  field.change();
  assert.equal(env.calls.length, 1);
  assert.equal(env.calls[0].data.amount, 200);
  env.calls[0].resolve({});
  await settle();
  assert.equal(field.values.dirty, false);
});

test('checkbox changes are immediate and an older reply cannot acknowledge newer edits', async () => {
  const env = environment(), field = env.bind();
  field.input('checkbox');
  assert.equal(env.calls.length, 1);
  field.values.amount = 2;
  field.input('checkbox');
  env.calls[0].resolve({});
  await settle();
  assert.equal(field.values.dirty, true);
  assert.equal(env.calls.length, 2);
  assert.equal(env.calls[1].data.amount, 2);
  env.calls[1].resolve({});
  await settle();
  assert.equal(field.values.dirty, false);
});

test('navigation preserves the old form request and its reply cannot mark a new form saved', async () => {
  const env = environment(), before = env.bind();
  before.values.amount = 12;
  before.input();
  before.form.isConnected = false;
  const after = env.bind();
  env.tick();
  assert.equal(env.calls[0].data.amount, 12);
  env.calls[0].resolve({});
  await settle();
  assert.equal(after.values.saved, 0);
});

test('an invalid edit discards an older queued value until the input is corrected', async () => {
  const env = environment(), active = env.bind('task/active'), waiting = env.bind('task/waiting');
  active.input('checkbox');
  waiting.input('checkbox');
  waiting.values.valid = false;
  waiting.input();
  env.tick();
  env.calls[0].resolve({});
  await settle();
  assert.equal(env.calls.length, 1);
  assert.equal(waiting.values.dirty, true);
  waiting.values.valid = true;
  waiting.values.amount = 3;
  waiting.input();
  env.tick();
  assert.equal(env.calls.length, 2);
  assert.equal(env.calls[1].data.amount, 3);
  env.calls[1].resolve({});
  await settle();
  assert.equal(waiting.values.dirty, false);
});

test('a failed request reports the error and the same value can be retried', async () => {
  const env = environment(), field = env.bind();
  field.input('checkbox');
  env.calls[0].reject(new Error('connection lost'));
  await settle();
  assert.match(env.errors[0], /connection lost/);
  assert.equal(field.values.dirty, true);
  field.change();
  assert.equal(env.calls.length, 2);
  env.calls[1].resolve({});
  await settle();
  assert.equal(field.values.dirty, false);
});
