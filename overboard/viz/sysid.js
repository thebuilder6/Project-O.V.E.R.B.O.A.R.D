/* OVERBOARD SysId - guided calibration: collect, measure, fit, compare.
 *
 * A standalone page: no field, no solve. It reuses the server's calibration
 * routes, so a session collected here fits with sysid_fit.py too. A pushed
 * trajectory session is compare-only and is never fitted here: the fit reads
 * calibration/<name>.* alone. Finish preselects the new fit in the main
 * Solve tab and never solves by itself. */

const el = (id) => document.getElementById(id);
const HANDOFF_KEY = 'overboard.sysid.handoff.v1';

const state = {
    jobId: null,
    pollTimer: null,
    sheet: { rows: [], weigh: [] },
    before: null,
    fitted: null,
    express: new Set(),
    anchor: null,
    profiles: [],
    // Step mode: the checked tests run as chained one-test pushes.
    step: null,
    prompts: [],
    progress: {},
    lastEnds: 0,
    lastLogLen: 0,
    lastLogAt: 0,
};

function toast(message, isError) {
    const box = el('sys-toast');
    box.textContent = message;
    box.hidden = false;
    if (isError) box.style.background = '#991b1b';
    else box.style.background = '';
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => { box.hidden = true; }, 4000);
}

async function api(path, options) {
    const response = await fetch(path, Object.assign(
        { headers: { 'Content-Type': 'application/json' } }, options || {}));
    const text = await response.text();
    let payload = null;
    try { payload = text ? JSON.parse(text) : null; } catch (err) { payload = null; }
    if (!response.ok) {
        const message = (payload && payload.error) || text || response.statusText;
        throw new Error(message);
    }
    return payload;
}

function robotName() {
    return (el('sys-name').value || 'robot').trim() || 'robot';
}

function slugify(name) {
    const slug = String(name || '').toLowerCase().replace(/[^\w-]+/g, '-')
        .replace(/^[-_]+|[-_]+$/g, '').slice(0, 60);
    return slug || 'robot';
}

/* The drivebase profile this collect pushes: the same example configs the
   main tab lists. The server syncs it to robot_config.py before pushing,
   so the hub's ports and geometry match the plan. */
async function loadProfiles() {
    try {
        const data = await api('/api/examples');
        state.profiles = (data && data.configs) || [];
    } catch (err) {
        state.profiles = [];
    }
    const select = el('sys-profile');
    select.innerHTML = '';
    state.profiles.forEach((config) => {
        const option = document.createElement('option');
        option.value = config.name;
        option.textContent = config.name;
        select.appendChild(option);
    });
    if (!state.profiles.length) {
        const option = document.createElement('option');
        option.value = 'my_robot';
        option.textContent = 'my_robot';
        select.appendChild(option);
    }
    const want = state.profiles.some((c) => c.name === 'my_robot') ? 'my_robot' : select.options[0].value;
    select.value = want;
    // Changing the profile renames the robot: the name is the stem of
    // calibration/<name>.*.
    el('sys-name').value = want;
}

function selectedProfileName() {
    const select = el('sys-profile');
    return (select && select.value) || 'my_robot';
}

/* ----------------------------------------------- 1b. the test list */

async function loadTests() {
    try {
        const bench = el('sys-bench').checked ? '?bench=1' : '';
        const [tests, sheet] = await Promise.all([
            api(`/api/calibration/tests${bench}`),
            api(`/api/calibration/sheet?name=${encodeURIComponent(robotName())}`),
        ]);
        state.express = new Set(tests.express || []);
        state.anchor = tests.anchor ? new Set(tests.anchor) : null;
        state.testPlan = (tests.tests || []).map((t) => ({ id: t.id, repeats: t.repeats || 1 }));
        renderTests(tests.tests || [], sheet, el('sys-bench').checked);
    } catch (err) {
        el('sys-tests-note').textContent = err.message;
    }
}

function testStatus(id, meta) {
    const runs = (meta || []).filter((entry) => entry.test === id);
    if (!runs.length) return 'missing';
    if (runs.some((entry) => !entry.ended || (entry.missing_rows || 0) > 0)) {
        return 'partial';
    }
    return 'done';
}

function renderTests(tests, sheet, bench) {
    const table = el('sys-tests');
    table.innerHTML = '';
    const header = document.createElement('tr');
    ['Run', 'Test', 'Repeats', 'Status', 'Setup first', 'What the robot does'].forEach((label) => {
        const th = document.createElement('th');
        th.textContent = label;
        header.appendChild(th);
    });
    table.appendChild(header);
    const meta = (sheet && sheet.meta) || [];
    tests.forEach((test) => {
        const tr = document.createElement('tr');
        const run = document.createElement('td');
        const box = document.createElement('input');
        box.type = 'checkbox';
        box.checked = true;
        box.dataset.test = test.id;
        box.addEventListener('change', updateTestNote);
        run.appendChild(box);
        tr.appendChild(run);
        const does = (bench && test.bench_skip)
            ? `${test.does} Skipped on a bench: the fit never reads these runs.`
            : test.does;
        [test.id + (test.deps && test.deps.length ? ` (needs ${test.deps.join(', ')})` : ''),
            `${test.repeats}x${test.tape ? ' + tape' : ''}${test.auto ? ', auto' : ''}`,
            testStatus(test.id, meta), test.setup || '', does || ''].forEach((text) => {
            const td = document.createElement('td');
            td.textContent = text;
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });
    const done = tests.filter((t) => testStatus(t.id, meta) === 'done').length;
    el('sys-tests-count').textContent =
        tests.length ? `${done} of ${tests.length} on file` : '';
    updateTestNote();
}

function selectedTests() {
    const boxes = Array.from(el('sys-tests').querySelectorAll('input[type=checkbox]'));
    if (!boxes.length) return null;
    const picked = boxes.filter((box) => box.checked).map((box) => box.dataset.test);
    // Everything picked is a fresh full suite (it starts the session over);
    // a proper subset appends and supersedes.
    const all = boxes.map((box) => box.dataset.test);
    if (picked.length === all.length) return null;
    return picked;
}

function updateTestNote() {
    const picked = selectedTests();
    let note = '';
    if (el('sys-bench').checked) {
        note = 'Bench mode: short runs for a small bench; the curvature tests are skipped on the hub. ';
    }
    if (picked !== null) {
        const tape = picked.filter((id) => !(state.express || new Set()).has(id));
        note += `${picked.length} selected; the rest stays as it is. `;
        if (!tape.length) note += 'Express: no tape needed, the wall is the anchor. ';
        else if (tape.length === 1 && tape[0] === 'odometry_straight') note += 'Anchor: wall primary, one tape check. ';
        else note += `Lab: ${tape.length} tape test(s), bring the tape measure. `;
        if (picked.includes('wall_stall') && !picked.includes('wall_homing')) {
            note += 'wall_homing runs too (stall needs the robot against the wall). ';
        }
    } else {
        note += 'Full suite (Lab): the session starts over. ';
    }
    el('sys-tests-note').textContent = note;
}

function presetTests(which) {
    const boxes = Array.from(el('sys-tests').querySelectorAll('input[type=checkbox]'));
    if (!boxes.length) return;
    const express = state.express || new Set();
    const anchor = state.anchor || null;
    boxes.forEach((box) => {
        if (which === 'lab') box.checked = true;
        else if (which === 'anchor' && anchor) box.checked = anchor.has(box.dataset.test);
        else if (which === 'anchor') box.checked = express.has(box.dataset.test) || box.dataset.test === 'odometry_straight';
        else box.checked = express.has(box.dataset.test);
    });
    updateTestNote();
}

/* ------------------------------------------------------------ 1. collect */

async function scanHubs() {
    el('sys-scan').disabled = true;
    el('sys-collect-status').textContent = 'Scanning Bluetooth (about a minute)...';
    try {
        const data = await api('/api/hubs', { method: 'POST', body: '{}' });
        const hubs = data.hubs || [];
        if (!hubs.length) {
            el('sys-collect-status').textContent =
                'No hubs seen. Turn the hub on and close Pybricks Code.';
        } else if (hubs.length === 1) {
            el('sys-hub').value = hubs[0].name || '';
            el('sys-collect-status').textContent = `Found one hub: ${hubs[0].label || hubs[0].name}.`;
        } else {
            el('sys-collect-status').textContent =
                `Found ${hubs.length} hubs; pick one and put its name in Hub name.`;
            toast(hubs.map((h) => h.label || h.name).join(' | '));
        }
    } catch (err) {
        el('sys-collect-status').textContent = err.message;
    }
    el('sys-scan').disabled = false;
}

async function startCollect() {
    const picked = selectedTests();
    if (picked !== null && !picked.length) {
        toast('Pick at least one test to collect.', true);
        return;
    }
    if (el('sys-step-mode').checked) {
        // Step mode: one push per checked test, chained. Setup steps wait
        // for the button as usual; setup-free singletons start by
        // themselves, since the Run press is the trigger already.
        const boxes = Array.from(el('sys-tests').querySelectorAll('input[type=checkbox]'));
        const queue = boxes.filter((box) => box.checked).map((box) => box.dataset.test);
        if (!queue.length) {
            toast('Pick at least one test to collect.', true);
            return;
        }
        state.step = { queue, total: queue.length };
        runNextStep();
        return;
    }
    collectTests(picked);
}

async function collectTests(picked) {
    state.lastEnds = 0;
    state.lastLogLen = 0;
    state.lastLogAt = Date.now();
    el('sys-log-details').open = true;
    el('sys-log-live').textContent = 'waiting for the hub…';
    el('sys-log-live').className = 'live quiet';
    el('sys-collect').disabled = true;
    el('sys-cancel').hidden = false;
    el('sys-collect-status').textContent = stepPrefix() + 'Pushing the test suite to the hub...';
    try {
        const body = {
            name: robotName(),
            connection: el('sys-conn').value || 'ble',
            hub_name: el('sys-hub').value || null,
            profile: selectedProfileName(),
        };
        if (picked !== null) body.tests = picked;
        if (el('sys-bench').checked) body.bench = true;
        const job = await api('/api/calibration/collect', {
            method: 'POST',
            body: JSON.stringify(body),
        });
        state.jobId = job.id;
        pollCollect();
    } catch (err) {
        el('sys-collect').disabled = false;
        el('sys-cancel').hidden = true;
        state.step = null;
        el('sys-collect-status').textContent = err.message;
        toast(err.message, true);
    }
}

function stepPrefix() {
    if (!state.step) return '';
    const done = state.step.total - state.step.queue.length;
    return `Step ${done + 1}/${state.step.total} (${state.step.queue[0]}): `;
}

function runNextStep() {
    if (!state.step || !state.step.queue.length) {
        state.step = null;
        return;
    }
    collectTests([state.step.queue[0]]);
}

async function cancelCollect() {
    if (!state.jobId) return;
    try {
        await api(`/api/jobs/${state.jobId}/cancel`, { method: 'POST', body: '{}' });
    } catch (err) {
        toast(err.message, true);
    }
}

/* The hub log is the proof the hub is alive: it opens itself while a
   collect runs, follows the tail, and says when the hub goes quiet. */
function renderLog(log, running) {
    const pre = el('sys-log');
    pre.textContent = (log || []).join('\n');
    const live = el('sys-log-live');
    if (running) {
        el('sys-log-details').open = true;
        if (log.length !== state.lastLogLen) {
            state.lastLogLen = log.length;
            state.lastLogAt = Date.now();
            // Follow the tail while it grows; the operator can still scroll
            // up, and the next lines pull it back down.
            pre.scrollTop = pre.scrollHeight;
        }
        const quiet_s = Math.floor((Date.now() - state.lastLogAt) / 1000);
        if (!log.length) {
            live.textContent = 'waiting for the hub…';
            live.className = 'live quiet';
        } else if (quiet_s > 10) {
            live.textContent = `${log.length} lines · quiet for ${quiet_s}s — check the hub`;
            live.className = 'live quiet';
        } else {
            live.textContent = `${log.length} lines · receiving…`;
            live.className = 'live';
        }
    } else if (log.length) {
        live.textContent = `${log.length} lines · finished`;
        live.className = 'live';
    } else {
        live.textContent = '';
        live.className = 'live';
    }
}

/* Last / now / on deck, from the plan and the hub's own prompt cards.
   Pure function of its inputs, so the order is checkable without a hub:
   plan is the suite order, checked is the table order (null = full suite),
   prompts are the hub's cards, current is progress.current, stepQueue is the
   step-mode queue. */
function queueTrio(plan, checked, prompts, current, stepQueue) {
    const order = checked === null ? (plan || []).map((t) => t.id) : (checked || []).slice();
    // The stall drags its homing along (see updateTestNote).
    if (order.includes('wall_stall') && !order.includes('wall_homing')) {
        order.splice(order.indexOf('wall_stall'), 0, 'wall_homing');
    }
    const repeatsOf = (id) => ((plan || []).find((t) => t.id === id) || {}).repeats || 1;
    const cards = prompts || [];
    const doneCards = cards.filter((c) => c.state === 'done');
    const runningCard = [...cards].reverse().find((c) => c.state !== 'done');
    const last = doneCards.length ? doneCards[doneCards.length - 1].id : '—';
    if (stepQueue && stepQueue.length) {
        return { last, now: stepQueue[0], ondeck: stepQueue.length > 1 ? stepQueue[1] : 'done' };
    }
    const cur = (runningCard && runningCard.id) || (current && current.test);
    if (!cur) {
        return { last, now: 'pushing…', ondeck: order.length ? order[0] : '—' };
    }
    const rep = current && current.repeat !== undefined && current.repeat !== null
        ? Number(current.repeat) : 0;
    if (rep + 1 < repeatsOf(cur)) {
        return { last, now: cur, ondeck: cur };
    }
    const idx = order.indexOf(cur);
    return { last, now: cur, ondeck: idx >= 0 && idx + 1 < order.length ? order[idx + 1] : 'done' };
}

function renderQueue(prompts, progress, finished) {
    const trio = finished
        ? { last: (prompts || []).filter((c) => c.state === 'done').slice(-1).map((c) => c.id)[0] || '—', now: 'finished', ondeck: '—' }
        : queueTrio(state.testPlan, selectedTests(), prompts, (progress || {}).current,
            state.step && state.step.queue);
    el('sys-queue').textContent = `Last: ${trio.last} · Now: ${trio.now} · On deck: ${trio.ondeck}`;
}

function pollCollect() {
    clearTimeout(state.pollTimer);
    state.pollTimer = setTimeout(async () => {
        try {
            const job = await api(`/api/jobs/${state.jobId}`);
            renderPrompts(job.prompts || [], job.progress || {});
            const ends = (job.prompts || []).filter((card) => card.state === 'done').length;
            if (ends !== state.lastEnds) {
                state.lastEnds = ends;
                await loadSheet();
                await loadTests();
                renderPrompts(job.prompts || [], job.progress || {});
            }
            const progress = job.progress || {};
            const current = progress.current;
            renderQueue(job.prompts || [], progress, job.status !== 'running');
            if (job.status === 'running') {
                renderLog(job.log || [], true);
                el('sys-collect-status').textContent = stepPrefix() + (current
                    ? `Running ${current.test}`
                        + `${current.repeat !== undefined && current.repeat !== null
                            ? `, repeat ${Number(current.repeat) + 1} of 3` : ''}. `
                        + `Press the centre button on the hub. `
                        + `${progress.needed || 0} measurements needed in total.`
                    : 'Pushing the test suite to the hub...');
                pollCollect();
                return;
            }
            el('sys-collect').disabled = false;
            el('sys-cancel').hidden = true;
            renderLog(job.log || [], false);
            if (job.status === 'ok') {
                el('sys-collect-status').textContent = stepPrefix() +
                    `Collected ${progress.tests_done || 0} tests. ` +
                    `Measure ${progress.needed || 0} poses with the tape, then fit.`;
                await loadSheet();
                await loadTests();
                if (state.step) {
                    state.step.queue.shift();
                    if (state.step.queue.length) {
                        runNextStep();
                    } else {
                        const total = state.step.total;
                        state.step = null;
                        toast(`Stepped through ${total} test${total === 1 ? '' : 's'}.`);
                    }
                }
            } else {
                state.step = null;
                el('sys-collect-status').textContent =
                    (job.error || 'The collection failed.')
                    + ' The session that arrived was kept.';
            }
        } catch (err) {
            el('sys-collect-status').textContent = err.message;
            el('sys-collect').disabled = false;
            el('sys-cancel').hidden = true;
        }
    }, 1000);
}

/* ------------------------------------------------------------ 2. measure */

function renderMeta(meta, flags) {
    const table = el('sys-meta');
    table.innerHTML = '';
    if (!meta || !meta.length) return;
    const header = document.createElement('tr');
    ['Test', 'Repeat', 'Battery (mV)', 'Wheel / track (mm)', 'Rows', 'Dropped'].forEach((label) => {
        const th = document.createElement('th');
        th.textContent = label;
        header.appendChild(th);
    });
    table.appendChild(header);
    meta.forEach((entry) => {
        const tr = document.createElement('tr');
        const cells = [
            entry.test || '',
            entry.repeat !== undefined && entry.repeat !== null ? String(entry.repeat) : '',
            entry.battery_mv !== undefined && entry.battery_mv !== null
                ? String(entry.battery_mv) : '',
            `${entry.wheel_diameter_mm || ''} / ${entry.axle_track_mm || ''}`,
            String(entry.rows || 0),
            entry.missing_rows ? `${entry.missing_rows} missing` : (entry.ended ? '' : 'cut short'),
        ];
        cells.forEach((text) => {
            const td = document.createElement('td');
            td.textContent = text;
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });
    const warn = el('sys-truncated');
    if (flags && flags.truncated) {
        warn.hidden = false;
        warn.textContent =
            `The link dropped part of the session (${flags.missing_rows || 0} rows missing). ` +
            `What arrived was kept whole; collect again for a complete run.`;
    } else {
        warn.hidden = true;
    }
    const volts = el('sys-voltages');
    const seen = [...new Set(meta.map((e) => e.battery_mv).filter((v) => v !== undefined && v !== null))];
    volts.textContent = seen.length
        ? `Hub voltage during this session: ${seen.join(', ')} mV.`
        : '';
}

/* One state, two views: the Measure table and the prompt cards render the
   same rows, so an entry box edits the state and every matching box
   follows. kind is 'measure' (test/repeat/stop/field) or 'weigh'
   (repeat). */
function syncMeasureInputs(kind, row, field, value) {
    document.querySelectorAll('input[data-kind]').forEach((other) => {
        if (other.dataset.kind !== kind) return;
        if (kind === 'weigh') {
            if (String(other.dataset.repeat) !== String(row.repeat)) return;
        } else if (other.dataset.test !== row.test
                || String(other.dataset.repeat) !== String(row.repeat)
                || String(other.dataset.stop) !== String(row.stop)
                || other.dataset.field !== field) {
            return;
        }
        if (other.value !== value) other.value = value;
    });
}

function bindSheetInput(input, kind, row, field) {
    if (kind === 'weigh') {
        input.dataset.kind = 'weigh';
        input.dataset.repeat = row.repeat;
    } else {
        input.dataset.kind = 'measure';
        input.dataset.test = row.test;
        input.dataset.repeat = row.repeat;
        input.dataset.stop = row.stop;
        input.dataset.field = field;
    }
    input.addEventListener('input', () => {
        row[field] = input.value;
        syncMeasureInputs(kind, row, field, input.value);
    });
}

function renderSheet(payload) {
    const table = el('sys-sheet');
    table.innerHTML = '';
    const rows = payload.rows || [];
    state.sheet.rows = rows;
    const tier = payload.tier || (payload.session ? 'lab' : 'none');
    const tierBox = el('sys-tier');
    if (tierBox) {
        tierBox.textContent = !payload.session
            ? 'No session yet: Express collects the no-tape tier first.'
            : tier === 'express'
                ? 'Express session: no tape needed, the wall is the anchor. Fit whenever ready.'
                : `Lab session: ${payload.total} tape measurements across the suite.`;
    }
    if (!rows.length) {
        el('sys-sheet-status').textContent = payload.session
            ? 'Express: nothing to measure with the tape. Fit the model, or add one tape anchor below.'
            : 'No measurements yet. Collect a session first.';
        el('sys-weigh').innerHTML = '';
        state.sheet.weigh = payload.weigh_in || [];
        renderMeta(payload.meta, payload);
        if (payload.geometry_warning) toast(payload.geometry_warning, true);
        return;
    }
    if (tier === 'anchor') {
        tierBox.textContent = 'Anchor session: wall primary, one tape check (odometry_straight). Fit, then promote to Lab only on disagreement.';
    }
    const header = document.createElement('tr');
    ['Test', 'Repeat', 'Stop', 'x (mm)', 'y (mm)', 'Turn (deg)'].forEach((label) => {
        const th = document.createElement('th');
        th.textContent = label;
        header.appendChild(th);
    });
    table.appendChild(header);
    rows.forEach((row) => {
        const tr = document.createElement('tr');
        [row.test, String(row.repeat), String(row.stop)].forEach((text) => {
            const td = document.createElement('td');
            td.textContent = text;
            tr.appendChild(td);
        });
        ['x_mm', 'y_mm', 'turn_right_deg'].forEach((field) => {
            const td = document.createElement('td');
            const input = document.createElement('input');
            input.type = 'number';
            input.step = 'any';
            input.value = row[field] === undefined || row[field] === null ? '' : row[field];
            bindSheetInput(input, 'measure', row, field);
            td.appendChild(input);
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });

    const weigh = el('sys-weigh');
    weigh.innerHTML = '';
    (payload.weigh_in || []).forEach((row) => {
        const tr = document.createElement('tr');
        const td = document.createElement('td');
        td.textContent = `repeat ${row.repeat}`;
        tr.appendChild(td);
        const cell = document.createElement('td');
        const input = document.createElement('input');
        input.type = 'number';
        input.step = 'any';
        input.value = row.mass_kg === undefined || row.mass_kg === null ? '' : row.mass_kg;
        bindSheetInput(input, 'weigh', row, 'mass_kg');
        cell.appendChild(input);
        const unit = document.createElement('span');
        unit.textContent = ' kg';
        cell.appendChild(unit);
        tr.appendChild(cell);
        weigh.appendChild(tr);
    });
    state.sheet.weigh = payload.weigh_in || [];

    const missing = payload.missing || 0;
    el('sys-sheet-status').textContent = missing
        ? `${missing} of ${payload.total} measurements still needed. Fill them in from your tape.`
        : `All ${payload.total} measurements are in. Fit the model.`;
    renderMeta(payload.meta, payload);
    if (payload.geometry_warning) toast(payload.geometry_warning, true);
}

async function loadSheet() {
    try {
        const payload = await api(
            `/api/calibration/sheet?name=${encodeURIComponent(robotName())}`);
        renderSheet(payload);
        el('sys-collect-count').textContent =
            payload.session ? `${payload.total} measurements` : '';
        await captureBefore();
    } catch (err) {
        el('sys-sheet-status').textContent = err.message;
    }
}

async function saveSheet() {
    const rows = (state.sheet.rows || []).map((row) => ({
        test: row.test,
        repeat: String(row.repeat),
        stop: String(row.stop),
        x_mm: row.x_mm === undefined || row.x_mm === null ? '' : row.x_mm,
        y_mm: row.y_mm === undefined || row.y_mm === null ? '' : row.y_mm,
        turn_right_deg: row.turn_right_deg === undefined || row.turn_right_deg === null
            ? '' : row.turn_right_deg,
    }));
    const weigh = (state.sheet.weigh || []).map((row) => ({
        test: 'weigh_in',
        repeat: row.repeat,
        stop: 0,
        mass_kg: row.mass_kg === undefined || row.mass_kg === null ? '' : row.mass_kg,
    }));
    try {
        const payload = await api('/api/calibration/sheet', {
            method: 'POST',
            body: JSON.stringify({ name: robotName(), rows, weigh_in: weigh }),
        });
        renderSheet(payload);
        renderPrompts(state.prompts, state.progress);
        toast('Saved the measurement sheet.');
    } catch (err) {
        toast(err.message, true);
    }
}

/* --------------------------------------------- the hub prompt feed */

function rowComplete(row) {
    return ['x_mm', 'y_mm', 'turn_right_deg'].every((field) =>
        row[field] !== '' && row[field] !== null && row[field] !== undefined);
}

function renderPrompts(prompts, progress) {
    state.prompts = prompts || [];
    state.progress = progress || {};
    const box = el('sys-prompts');
    box.innerHTML = '';
    if (!state.prompts.length) {
        const hint = document.createElement('p');
        hint.className = 'hint';
        hint.textContent = 'No run yet: each hub prompt appears here as it runs, '
            + 'with entry boxes for the numbers it asked for.';
        box.appendChild(hint);
        return;
    }
    state.prompts.forEach((card) => {
        const article = document.createElement('article');
        article.className = 'prompt-card';
        const head = document.createElement('div');
        head.className = 'prompt-head';
        const name = document.createElement('strong');
        name.textContent = card.id;
        head.appendChild(name);
        if (card.title) {
            const title = document.createElement('span');
            title.className = 'prompt-title';
            title.textContent = card.title;
            head.appendChild(title);
        }
        const badge = document.createElement('span');
        badge.className = 'badge' + (card.state === 'done' ? ' done' : '');
        badge.textContent = card.state === 'running' ? 'running'
            : card.state === 'measuring' ? 'measuring — get the tape ready'
            : 'done';
        head.appendChild(badge);
        article.appendChild(head);
        if (card.setup) {
            const setup = document.createElement('p');
            setup.className = 'prompt-setup';
            setup.textContent = card.setup;
            article.appendChild(setup);
        }
        const current = (state.progress && state.progress.current) || {};
        if (card.state === 'running' && current.test === card.id) {
            const repeat = document.createElement('p');
            repeat.className = 'hint';
            repeat.textContent = current.repeat !== undefined && current.repeat !== null
                ? `Repeat ${Number(current.repeat) + 1} of 3.` : 'Running.';
            article.appendChild(repeat);
        }
        const rows = (state.sheet.rows || []).filter((row) => row.test === card.id);
        if (card.state !== 'running') {
            if (!rows.length) {
                const note = document.createElement('p');
                note.className = 'hint';
                note.textContent = 'Watch-only: nothing to measure for this one.';
                article.appendChild(note);
            } else {
                const grid = document.createElement('div');
                grid.className = 'prompt-rows';
                rows.forEach((row) => {
                    const key = document.createElement('span');
                    key.className = 'prompt-key';
                    key.textContent = `r${row.repeat} s${row.stop}`;
                    grid.appendChild(key);
                    ['x_mm', 'y_mm', 'turn_right_deg'].forEach((field) => {
                        const input = document.createElement('input');
                        input.type = 'number';
                        input.step = 'any';
                        input.title = `${row.test} repeat ${row.repeat} stop ${row.stop} ${field}`;
                        input.value = row[field] === undefined || row[field] === null
                            ? '' : row[field];
                        bindSheetInput(input, 'measure', row, field);
                        grid.appendChild(input);
                    });
                });
                article.appendChild(grid);
                const done = rows.every(rowComplete);
                const row = document.createElement('div');
                row.className = 'row';
                const save = document.createElement('button');
                save.type = 'button';
                save.className = 'btn small';
                save.textContent = done ? 'Saved ✓' : 'Save these numbers';
                save.addEventListener('click', saveSheet);
                row.appendChild(save);
                article.appendChild(row);
            }
        }
        box.appendChild(article);
    });
}

/* ---------------------------------------------------------------- 3. fit */

function fmt(value, digits) {
    return (value === undefined || value === null) ? '—' : Number(value).toFixed(digits);
}

function fmtCi(ci, digits) {
    if (!ci || ci.length !== 2) return '';
    return ` [${Number(ci[0]).toFixed(digits)}, ${Number(ci[1]).toFixed(digits)}]`;
}

async function captureBefore() {
    // The "before" is the prior fit for this robot, read through the same
    // summarize_fit reader the "after" uses. None yet means the nominal
    // robot config is the before.
    state.before = null;
    try {
        const data = await api('/api/robot/models');
        const want = slugify(robotName());
        const found = (data.models || []).find((model) => model.name === want);
        if (found) state.before = found;
    } catch (err) {
        // No measured model yet; the nominal one is the before.
    }
    el('sys-before-note').textContent = state.before
        ? `Before: ${state.before.name} (${state.before.source}).`
        : 'Before: no prior fit for this robot. The nominal robot config is the before.';
}

async function fitModel() {
    await captureBefore();
    const payload = { name: robotName() };
    const gearing = Number(el('sys-gearing').value.trim());
    if (el('sys-gearing').value.trim() !== '' && isFinite(gearing) && gearing > 0) {
        payload.gearing = gearing;
    }
    const wall = Number(el('sys-wall').value.trim());
    if (el('sys-wall').value.trim() !== '' && isFinite(wall) && wall > 0) {
        payload.wall_distance_m = wall;
    }
    el('sys-fit').disabled = true;
    try {
        const summary = await api('/api/calibration/fit', {
            method: 'POST',
            body: JSON.stringify(payload),
        });
        state.fitted = summary;
        const problems = summary.problems || [];
        const landed = [];
        if (summary.mass && summary.mass.calibrated) {
            landed.push(`mass ${fmt(summary.mass.mass_kg, 4)} kg`);
        }
        if (summary.inertia && summary.inertia.calibrated) {
            landed.push(`inertia ${fmt(summary.inertia.inertia_kg_m2, 6)} kg m2`);
        }
        const tier = summary.tier || 'lab';
        const tierLine = tier === 'express'
            ? 'Express fit: dynamics only, odometry/drift/slip stay uncalibrated. '
            : tier === 'anchor'
                ? 'Anchor fit: wall primary, one tape check. Promote to Lab only on disagreement. '
                : 'Lab fit: tape scales included. ';
        el('sys-fit-status').textContent = problems.length
            ? `${tierLine}Fitted with warnings: ${problems.join('; ')}` + (landed.length ? ` Landed: ${landed.join(', ')}.` : '')
            : `${tierLine}Fitted. The model is saved as ${summary.path}.` + (landed.length ? ` Landed: ${landed.join(', ')}.` : '');
        if (summary.missing_rows || summary.bad_rows) {
            el('sys-fit-status').textContent +=
                ` Session gaps: ${summary.missing_rows || 0} missing, ${summary.bad_rows || 0} bad rows.`;
        }
        renderSummary(summary);
        renderCompare(state.before, summary);
        el('sys-finish').disabled = false;
        const anchor = el('sys-anchor');
        const anchorNote = el('sys-anchor-note');
        if (tier === 'express') {
            anchor.hidden = false;
            anchorNote.hidden = false;
            anchorNote.textContent = 'One tape anchor checks the wall scale: collects odometry_straight (3 runs), then enter its 3 tape poses and fit again.';
        } else {
            anchor.hidden = true;
            anchorNote.hidden = true;
        }
        toast('The measured model is ready.');
    } catch (err) {
        el('sys-fit-status').textContent = err.message;
        toast(err.message, true);
    }
    el('sys-fit').disabled = false;
}

async function addAnchor() {
    // Anchor as an option: one tape check on top of Express. Appends
    // odometry_straight (3 repeats) to the session; the sheet then shows
    // its 3 rows, the tape checks the wall scale, and the next fit is
    // Anchor (Lab only on disagreement).
    el('sys-anchor').disabled = true;
    try {
        collectTests(['odometry_straight']);
    } finally {
        el('sys-anchor').disabled = false;
    }
}

function renderSummary(summary) {
    const facts = el('sys-summary');
    facts.innerHTML = '';
    const items = [
        ['Straight gain', `${fmt(summary.straight && summary.straight.gain, 3)}${fmtCi(summary.straight && summary.straight.gain_ci95, 3)}`],
        ['Straight tau / delay', `${fmt(summary.straight && summary.straight.tau_s, 3)} s / ${fmt(summary.straight && summary.straight.delay_s, 3)} s`],
        ['Turn gain', `${fmt(summary.turn && summary.turn.gain, 3)}${fmtCi(summary.turn && summary.turn.gain_ci95, 3)}`],
        ['Turn tau / delay', `${fmt(summary.turn && summary.turn.tau_s, 3)} s / ${fmt(summary.turn && summary.turn.delay_s, 3)} s`],
        ['Gyro bias', `${fmt(summary.gyro_z_bias_deg_s, 4)} deg/s`],
        ['Distance scale', `${fmt(summary.odometry && summary.odometry.distance_scale, 4)} (${(summary.odometry && summary.odometry.distance_scale_source) || 'default'})${fmtCi(summary.odometry && summary.odometry.distance_scale_ci95, 4)}`],
        ['Heading scale', `${fmt(summary.odometry && summary.odometry.heading_scale, 4)}${fmtCi(summary.odometry && summary.odometry.heading_scale_ci95, 4)}`],
        ['Brake slip', summary.slip && summary.slip.calibrated
            ? `median ${fmt(summary.slip.brake_ratio_median, 3)} [${fmt(summary.slip.brake_ratio_min, 3)}, ${fmt(summary.slip.brake_ratio_max, 3)}]` : 'not measured'],
        ['Drift', summary.drift && summary.drift.calibrated
            ? `${fmt(summary.drift.drift_deg_per_m, 3)} deg/m${fmtCi(summary.drift.drift_deg_per_m_ci95, 3)} (tape ${fmt(summary.drift.drift_deg_per_m_tape, 3)})` : 'not measured'],
        ['Stall torque', summary.stall && summary.stall.calibrated
            ? `${fmt(summary.stall.t_max_nm, 4)} Nm${fmtCi(summary.stall.t_max_nm_ci95, 4)}` : 'not measured'],
        ['No-load speed', summary.top_speed && summary.top_speed.calibrated
            ? `${fmt(summary.top_speed.no_load_speed_rad_s, 2)} rad/s${fmtCi(summary.top_speed.no_load_speed_rad_s_ci95, 2)}` : 'not measured'],
        ['Mass', summary.mass && summary.mass.calibrated
            ? `${fmt(summary.mass.mass_kg, 4)} kg${fmtCi(summary.mass.mass_kg_ci95, 4)}` : 'not weighed'],
        ['Inertia', summary.inertia && summary.inertia.calibrated
            ? `${fmt(summary.inertia.inertia_kg_m2, 6)} kg m2${fmtCi(summary.inertia.inertia_kg_m2_ci95, 6)}` : 'not derived'],
    ];
    items.forEach(([label, text]) => {
        const dt = document.createElement('dt');
        dt.textContent = label;
        const dd = document.createElement('dd');
        dd.textContent = text;
        facts.appendChild(dt);
        facts.appendChild(dd);
    });
}

/* ------------------------------------------------------ 4. refinement */

function compareRows(before, after) {
    const rows = [];
    const pair = (label, get, digits) => {
        const a = before ? get(before) : null;
        const b = get(after);
        rows.push([label, a === null || a === undefined ? 'nominal' : Number(a).toFixed(digits),
            b === null || b === undefined ? '—' : Number(b).toFixed(digits)]);
    };
    pair('Straight gain', (s) => s.straight && s.straight.gain, 3);
    pair('Straight tau (s)', (s) => s.straight && s.straight.tau_s, 3);
    pair('Straight delay (s)', (s) => s.straight && s.straight.delay_s, 3);
    pair('Turn gain', (s) => s.turn && s.turn.gain, 3);
    pair('Turn tau (s)', (s) => s.turn && s.turn.tau_s, 3);
    pair('Turn delay (s)', (s) => s.turn && s.turn.delay_s, 3);
    pair('Gyro bias (deg/s)', (s) => s.gyro_z_bias_deg_s, 4);
    pair('Distance scale', (s) => s.odometry && s.odometry.distance_scale, 4);
    pair('Heading scale', (s) => s.odometry && s.odometry.heading_scale, 4);
    pair('Brake slip median', (s) => s.slip && s.slip.brake_ratio_median, 3);
    pair('Drift (deg/m)', (s) => s.drift && s.drift.drift_deg_per_m, 3);
    pair('Stall torque (Nm)', (s) => s.stall && s.stall.t_max_nm, 4);
    pair('No-load speed (rad/s)', (s) => s.top_speed && s.top_speed.no_load_speed_rad_s, 2);
    pair('Mass (kg)', (s) => s.mass && s.mass.mass_kg, 4);
    pair('Inertia (kg m2)', (s) => s.inertia && s.inertia.inertia_kg_m2, 6);
    pair('Wheel radius (m)', (s) => s.wheel_radius_m, 4);
    return rows;
}

function renderCompare(before, after) {
    const table = el('sys-compare');
    table.innerHTML = '';
    const header = document.createElement('tr');
    ['Parameter', 'Before', 'After'].forEach((label) => {
        const th = document.createElement('th');
        th.textContent = label;
        header.appendChild(th);
    });
    table.appendChild(header);
    compareRows(before, after).forEach(([label, a, b]) => {
        const tr = document.createElement('tr');
        [label, a, b].forEach((text) => {
            const td = document.createElement('td');
            td.textContent = text;
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });
}

function finish() {
    // No solve here. Preselect the new fit in the main Solve tab: the studio
    // reads this handoff on load, ticks "Plan against the measured model"
    // and runs nothing until the operator presses Solve.
    if (!state.fitted) return;
    try {
        localStorage.setItem(HANDOFF_KEY, JSON.stringify({
            path: state.fitted.path,
            name: state.fitted.name || robotName(),
        }));
    } catch (err) {
        // Storage blocked: the query carries the path instead.
    }
    const path = state.fitted.path || '';
    location.href = path ? `/?sysid_fit=${encodeURIComponent(path)}` : '/';
}

el('sys-scan').addEventListener('click', scanHubs);
el('sys-collect').addEventListener('click', startCollect);
el('sys-cancel').addEventListener('click', cancelCollect);
el('sys-copy-log').addEventListener('click', () => {
    const text = el('sys-log').textContent || '';
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(
            () => toast('The log is on the clipboard.'),
            () => toast('Copy refused; select the log by hand.', true));
    } else {
        const range = document.createRange();
        range.selectNodeContents(el('sys-log'));
        const selection = getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        toast('Log selected; press Ctrl+C.');
    }
});
el('sys-reload').addEventListener('click', loadSheet);
el('sys-save').addEventListener('click', saveSheet);
el('sys-fit').addEventListener('click', fitModel);
el('sys-anchor').addEventListener('click', addAnchor);
el('sys-finish').addEventListener('click', finish);
el('sys-express').addEventListener('click', () => presetTests('express'));
el('sys-anchor-pick').addEventListener('click', () => presetTests('anchor'));
el('sys-lab').addEventListener('click', () => presetTests('lab'));
el('sys-select-all').addEventListener('click', () => {
    el('sys-tests').querySelectorAll('input[type=checkbox]')
        .forEach((box) => { box.checked = true; });
    updateTestNote();
});
el('sys-select-none').addEventListener('click', () => {
    el('sys-tests').querySelectorAll('input[type=checkbox]')
        .forEach((box) => { box.checked = false; });
    updateTestNote();
});
el('sys-name').addEventListener('change', () => { state.before = null; loadSheet(); loadTests(); });
el('sys-profile').addEventListener('change', () => {
    el('sys-name').value = selectedProfileName();
    state.before = null; loadSheet(); loadTests();
});
el('sys-bench').addEventListener('change', loadTests);

loadProfiles().then(() => { loadSheet(); loadTests(); });
