/* OVERBOARD Studio - draw a mission, solve it, download the result. */

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const el = (id) => document.getElementById(id);

const state = {
    config: null,
    configName: '',
    waypoints: [],
    selected: -1,
    trajectory: null,
    showEnvelope: true,
    showPath: true,
    jobId: null,
    pollTimer: null,
    health: null,
    presets: {},
    dirty: false,
    savedName: '',
    configEdited: false,
    failingSegments: new Set(),
    worstSegment: -1,
    reportCandidate: null,
};

/* Undo/redo history: a stack of mission snapshots. */
const history = { stack: [], index: -1, limit: 100 };

/* The mission is kept in the browser so a reload never loses the work. */
const SESSION_KEY = 'overboard.studio.session.v1';
let sessionTimer = null;

/* Camera: pixels per metre plus a device-pixel offset. */
const camera = { scale: 400, offsetX: 0, offsetY: 0 };
const playback = { time: 0, playing: false, duration: 0 };

/* ------------------------------------------------------------------ api */

const SERVER_GONE = 'Lost contact with the studio server. It may have stopped - '
    + 'restart it with: python main.py --gui';

async function api(path, options = {}) {
    let response;
    try {
        response = await fetch(path, {
            headers: { 'Content-Type': 'application/json' },
            ...options,
        });
    } catch (err) {
        throw new Error(SERVER_GONE);
    }
    let payload = null;
    try {
        payload = await response.json();
    } catch (err) {
        payload = null;
    }
    if (!response.ok) {
        if (payload && payload.error) throw new Error(payload.error);
        // A gateway error means nothing is listening on the studio port.
        if (response.status >= 500) throw new Error(SERVER_GONE);
        throw new Error(`${response.status} ${response.statusText}`);
    }
    return payload;
}

function toast(message, isError = false) {
    const node = el('toast');
    node.textContent = message;
    node.classList.toggle('error', isError);
    node.hidden = false;
    clearTimeout(node._timer);
    node._timer = setTimeout(() => { node.hidden = true; }, isError ? 9000 : 4000);
}

function setStatus(text, kind = '') {
    const node = el('header-status');
    node.textContent = text;
    node.className = `status ${kind}`;
}

/* -------------------------------------------------------------- mission */

function blankWaypoint(x = 0, y = 0, heading = 0) {
    return { x, y, heading, stop: false, event: '', precision_mode: false, error_threshold: null };
}

function setWaypoints(points) {
    state.waypoints = points.map((point) => {
        const heading = (point.heading === null || point.heading === undefined)
            ? null : Number(point.heading);
        return {
            x: Number(point.x),
            y: Number(point.y),
            heading: Number.isFinite(heading) ? heading : null,
            stop: Boolean(point.stop),
            event: point.event || '',
            precision_mode: Boolean(point.precision_mode),
            error_threshold: point.error_threshold === undefined || point.error_threshold === null
                ? null : Number(point.error_threshold),
        };
    });
    state.selected = state.waypoints.length ? 0 : -1;
    state.dirty = false;
    state.savedName = '';
    renderAll();
    pushHistory();
}

function resetHistory() {
    history.stack = [snapshot()];
    history.index = 0;
    updateHistoryButtons();
}

function waypointToJson(point) {
    const out = { x: point.x, y: point.y, heading: point.heading };
    if (point.stop) out.stop = true;
    if (point.event) out.event = point.event;
    if (point.precision_mode) out.precision_mode = true;
    if (point.error_threshold !== null) out.error_threshold = point.error_threshold;
    return out;
}

function currentWaypoint() {
    if (state.selected < 0 || state.selected >= state.waypoints.length) return null;
    return state.waypoints[state.selected];
}

/* --------------------------------------------------------- history */

function snapshot() {
    return {
        waypoints: JSON.parse(JSON.stringify(state.waypoints)),
        selected: state.selected,
    };
}

/* Record the current mission so it can be brought back. */
function pushHistory() {
    const current = history.stack[history.index];
    const now = snapshot();
    if (current && JSON.stringify(current.waypoints) === JSON.stringify(now.waypoints)) {
        return;
    }
    history.stack = history.stack.slice(0, history.index + 1);
    history.stack.push(now);
    if (history.stack.length > history.limit) history.stack.shift();
    history.index = history.stack.length - 1;
    updateHistoryButtons();
    scheduleSessionSave();
}

function applySnapshot(snap) {
    state.waypoints = JSON.parse(JSON.stringify(snap.waypoints));
    state.selected = Math.min(snap.selected, state.waypoints.length - 1);
    state.dirty = true;
    renderAll();
    draw();
    scheduleSessionSave();
}

function undo() {
    if (history.index <= 0) return;
    history.index -= 1;
    applySnapshot(history.stack[history.index]);
    updateHistoryButtons();
}

function redo() {
    if (history.index >= history.stack.length - 1) return;
    history.index += 1;
    applySnapshot(history.stack[history.index]);
    updateHistoryButtons();
}

function updateHistoryButtons() {
    el('btn-undo').disabled = history.index <= 0;
    el('btn-redo').disabled = history.index >= history.stack.length - 1;
}

/* ------------------------------------------------- session in the browser */

function scheduleSessionSave() {
    clearTimeout(sessionTimer);
    sessionTimer = setTimeout(saveSession, 300);
}

function saveSession() {
    try {
        localStorage.setItem(SESSION_KEY, JSON.stringify({
            name: el('mission-name').value,
            waypoints: state.waypoints,
            selected: state.selected,
            config: state.config,
            configName: state.configName,
            configEdited: state.configEdited,
            options: collectOptions(),
        }));
    } catch (err) {
        // Storage can be full or blocked; the mission is still undoable.
    }
}

function loadSession() {
    try {
        const raw = localStorage.getItem(SESSION_KEY);
        return raw ? JSON.parse(raw) : null;
    } catch (err) {
        return null;
    }
}

function clearSession() {
    try {
        localStorage.removeItem(SESSION_KEY);
    } catch (err) {
        // nothing to do
    }
}

function applyOptionsToForm(options) {
    if (!options) return;
    if (options.samples !== undefined) el('opt-samples').value = options.samples;
    if (options.accuracy_weight !== undefined) el('opt-accuracy').value = options.accuracy_weight;
    if (options.candidates !== undefined) el('opt-candidates').value = options.candidates;
    if (options.timing !== undefined) el('opt-timing').value = options.timing;
    if (options.validate !== undefined) el('opt-validate').checked = options.validate;
    if (options.export_format !== undefined) el('opt-export').value = options.export_format;
    if (options.controller_dt !== undefined) el('opt-controller-dt').value = options.controller_dt;
    updateCandidatesHint();
    syncPresetRadio();
}

/* Show the preset that matches the current options, if there is one. */
function syncPresetRadio() {
    const current = collectOptions();
    const radios = Array.from(document.querySelectorAll('#presets input[type=radio]'));
    const match = radios.find((radio) => {
        const preset = state.presets[radio.value];
        return preset
            && Number(preset.samples) === Number(current.samples)
            && Number(preset.accuracy_weight) === Number(current.accuracy_weight)
            && Number(preset.candidates) === Number(current.candidates)
            && Boolean(preset.validate) === Boolean(current.validate)
            && preset.export_format === current.export_format;
    });
    radios.forEach((radio) => { radio.checked = match ? radio.value === match.value : false; });
}

/* ------------------------------------------------------ mission files */

function downloadMission() {
    if (state.waypoints.length < 2) {
        toast('A mission needs at least two waypoints.', true);
        return;
    }
    const name = (el('mission-name').value || 'mission').replace(/[^a-z0-9_-]+/gi, '-');
    const blob = new Blob([JSON.stringify(state.waypoints.map(waypointToJson), null, 1)],
        { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `${name}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    toast(`Downloaded ${name}.json`);
}

/* The CLI prints a machine-readable validation summary on one line. */
const VALIDATION_MARKER = '##OVERBOARD-VALIDATION## ';

function parseValidationReport(logLines) {
    for (let i = (logLines || []).length - 1; i >= 0; i--) {
        const at = logLines[i].indexOf(VALIDATION_MARKER);
        if (at < 0) continue;
        try {
            return JSON.parse(logLines[i].slice(at + VALIDATION_MARKER.length));
        } catch (err) {
            return null;
        }
    }
    return null;
}

function renderValidationReport(report) {
    const box = el('validation-report');
    if (!report || !report.candidates || !report.candidates.length) {
        box.hidden = true;
        state.failingSegments = new Set();
        state.worstSegment = -1;
        return;
    }
    box.hidden = false;

    const index = state.reportCandidate !== null && state.reportCandidate < report.candidates.length
        ? state.reportCandidate
        : (report.selected || 1) - 1;
    const candidate = report.candidates[index] || report.candidates[0];

    // One chip per candidate, so a failed run can be compared.
    const chips = el('validation-candidates');
    chips.innerHTML = '';
    report.candidates.forEach((entry, i) => {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = `chip ${entry.ok ? 'ok' : 'bad'}${i === index ? ' active' : ''}`;
        const timing = entry.timing === 'per-segment' ? 'per-seg' : 'uniform';
        chip.textContent = `${i + 1}. ${entry.style || 'baseline'}, ${timing} - ${entry.ok ? 'ok' : 'failed'}`;
        chip.addEventListener('click', () => {
            state.reportCandidate = i;
            renderValidationReport(report);
        });
        chips.appendChild(chip);
    });

    const table = el('validation-table');
    table.innerHTML = '';
    const header = document.createElement('tr');
    ['Segment', 'Envelope', 'Limit', 'Drift'].forEach((label) => {
        const th = document.createElement('th');
        th.textContent = label;
        header.appendChild(th);
    });
    table.appendChild(header);

    const failing = new Set();
    let worst = -1;
    let worstDrift = -1;
    (candidate.segments || []).forEach((segment) => {
        const row = document.createElement('tr');
        const drift = (segment.drift_m || 0) * 100;
        const envelope = (segment.envelope_m || 0) * 100;
        const limit = (segment.limit_m || 0) * 100;
        if (!segment.ok) failing.add(segment.from);
        if (drift > worstDrift) { worstDrift = drift; worst = segment.from; }
        row.innerHTML = `<td>W${segment.from} &rarr; W${segment.to}</td>`
            + `<td>${envelope.toFixed(2)} cm</td>`
            + `<td>${limit.toFixed(2)} cm</td>`
            + `<td>${drift.toFixed(2)} cm</td>`;
        if (!segment.ok) row.className = 'bad';
        else if (worst === segment.from && worstDrift > 0.005) row.className = 'warn';
        table.appendChild(row);
    });
    state.failingSegments = failing;
    state.worstSegment = worst;

    const notes = [];
    if (candidate.tracking_cm !== null && candidate.tracking_cm !== undefined) {
        notes.push(`closed-loop tracking ${candidate.tracking_cm.toFixed(2)} cm (gate 0.50 cm)`);
    }
    if (candidate.reasons && candidate.reasons.length) {
        notes.push(candidate.reasons.join('; '));
    }
    if (worst >= 0 && worstDrift > 0.005) {
        notes.push(`the path drifts most between W${worst} and W${worst + 1}`);
    }
    el('validation-note').textContent = notes.join('. ');
}

/* ------------------------------------------------------------- rendering */

function renderAll() {
    renderWaypointList();
    renderEditor();
    renderResult();
    updateWaypointCount();
    updateMissionState();
}

function updateWaypointCount() {
    const count = state.waypoints.length;
    el('waypoint-count').textContent = count === 0
        ? 'No waypoints yet'
        : `${count} waypoint${count === 1 ? '' : 's'}`;
}

function updateMissionState() {
    const node = el('mission-state');
    if (!state.savedName) {
        node.textContent = 'Not saved to missions/ yet';
        node.className = 'hint';
    } else if (state.dirty) {
        node.textContent = `Changed since missions/${state.savedName}.json`;
        node.className = 'hint warn';
    } else {
        node.textContent = `Saved as missions/${state.savedName}.json`;
        node.className = 'hint';
    }
}

function renderWaypointList() {
    const list = el('wp-list');
    list.innerHTML = '';
    state.waypoints.forEach((point, index) => {
        const item = document.createElement('li');
        if (index === state.selected) item.classList.add('active');
        const tags = [];
        if (point.stop) tags.push('stop');
        if (point.event) tags.push(point.event);
        if (point.precision_mode) tags.push('precision');
        item.innerHTML = `<span>W${index} <span class="mono">(${point.x.toFixed(2)}, ${point.y.toFixed(2)})</span></span>`
            + `<span class="tags">${tags.join(', ')}</span>`;
        item.addEventListener('click', () => {
            state.selected = index;
            renderWaypointList();
            renderEditor();
        });
        list.appendChild(item);
    });
}

function renderEditor() {
    const point = currentWaypoint();
    el('no-selection').hidden = Boolean(point);
    el('waypoint-editor').hidden = !point;
    if (!point) return;

    el('wp-x').value = point.x.toFixed(3);
    el('wp-y').value = point.y.toFixed(3);
    const free = point.heading === null;
    el('wp-free-heading').checked = free;
    el('wp-heading').disabled = free;
    el('wp-heading').value = free ? '' : (point.heading * 180 / Math.PI).toFixed(1);
    el('wp-stop').checked = point.stop;
    el('wp-event').value = point.event || '';
    el('wp-precision').checked = point.precision_mode;
    el('wp-threshold').value = point.error_threshold === null
        ? '' : (point.error_threshold * 100).toFixed(2);
}

function renderResult() {
    const facts = el('result-facts');
    facts.innerHTML = '';
    const summary = state.jobSummary;
    if (!summary) return;
    const rows = [
        ['Duration', `${summary.duration_s.toFixed(3)} s`],
        ['Path', `${summary.path_length_m.toFixed(3)} m`],
        ['Peak wheel speed', `${summary.peak_wheel_speed_m_s.toFixed(2)} m/s`],
        ['Samples', String(summary.samples)],
    ];
    if (summary.has_envelope) {
        rows.push(['Max envelope', `${(summary.max_envelope_m * 100).toFixed(2)} cm`]);
    }
    rows.forEach(([key, value]) => {
        const dt = document.createElement('dt');
        dt.textContent = key;
        const dd = document.createElement('dd');
        dd.textContent = value;
        facts.appendChild(dt);
        facts.appendChild(dd);
    });
}

function renderDownloads(artifacts) {
    const box = el('downloads');
    box.innerHTML = '';
    const labels = {
        traj: 'Trajectory (.traj)',
        controller: 'Controller (.json)',
        python: 'Python module',
        stats: 'Stats (.json)',
    };
    (artifacts || []).forEach((artifact) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'btn small';
        button.textContent = labels[artifact.kind] || artifact.kind;
        button.addEventListener('click', () => {
            window.location.href = `/api/jobs/${state.jobId}/artifact/${artifact.kind}`;
        });
        box.appendChild(button);
    });
}

function renderRobotFacts() {
    const facts = el('robot-facts');
    facts.innerHTML = '';
    const robot = (state.config && (state.config.robot || state.config)) || {};
    const rows = [
        ['Mass', robot.mass !== undefined ? `${robot.mass} kg` : '-'],
        ['Wheel radius', robot.wheel_radius !== undefined ? `${(robot.wheel_radius * 1000).toFixed(1)} mm` : '-'],
        ['Track width', robot.track_width !== undefined ? `${(robot.track_width * 1000).toFixed(1)} mm` : '-'],
        ['Friction', robot.cof !== undefined ? String(robot.cof) : '-'],
    ];
    rows.forEach(([key, value]) => {
        const dt = document.createElement('dt');
        dt.textContent = key;
        const dd = document.createElement('dd');
        dd.textContent = value;
        facts.appendChild(dt);
        facts.appendChild(dd);
    });

    const warning = el('robot-warning');
    const isExample = state.configName === 'fll_choreo' && !state.configEdited;
    warning.hidden = !isExample && !state.configEdited;
    if (isExample) {
        warning.textContent = 'This is the example robot config. Its uncertainty ranges are test '
            + 'values, not measurements of your robot.';
    } else if (state.configEdited) {
        warning.textContent = 'Edited in the studio. Check every value against your robot, '
            + 'including the uncertainty ranges.';
    }
}

/* ------------------------------------------------------------ robot editor */

/* The fields a team actually measures. Lengths are shown in millimetres
 * because that is how a robot is measured; the config keeps metres. */
const ROBOT_FIELDS = [
    { key: 'mass', label: 'Mass', unit: 'kg', step: 0.001, min: 0.05 },
    { key: 'inertia', label: 'Inertia', unit: 'kg m2', step: 0.0001, min: 0 },
    { key: 'wheel_radius', label: 'Wheel radius', unit: 'mm', scale: 1000, step: 0.5, min: 1 },
    { key: 'track_width', label: 'Track width', unit: 'mm', scale: 1000, step: 0.5, min: 1 },
    { key: 'v_max_rad_s', label: 'No-load speed', unit: 'rad/s', step: 0.1, min: 0.1 },
    { key: 't_max_nm', label: 'Stall torque', unit: 'N m', step: 0.001, min: 0 },
    { key: 'gearing', label: 'Gearing', unit: 'ratio', step: 0.01, min: 0.01 },
    { key: 'cof', label: 'Friction', unit: 'cof', step: 0.01, min: 0.05 },
    { key: 'torque_headroom', label: 'Torque headroom', unit: 'ratio', step: 0.01, min: 0.05, max: 1 },
    { key: 'speed_headroom', label: 'Speed headroom', unit: 'ratio', step: 0.01, min: 0.05, max: 1 },
];

const RANGE_FIELDS = [
    { key: 'cof', label: 'Friction' },
    { key: 'torque_headroom', label: 'Torque headroom' },
    { key: 'gear_backlash_m', label: 'Gear backlash', unit: 'mm', scale: 1000 },
    { key: 'track_width_range', label: 'Track width', unit: 'mm', scale: 1000 },
    { key: 'wheel_radius_range', label: 'Wheel radius', unit: 'mm', scale: 1000 },
    { key: 'inertia_kg_m2', label: 'Inertia' },
];

/* The robot fields of a config in any of the three accepted layouts. */
function robotFields() {
    const config = state.config || {};
    const source = config.robot || config.config || config;
    const fields = {};
    Object.keys(source).forEach((key) => {
        const value = source[key];
        if (value && typeof value === 'object' && !Array.isArray(value) && 'val' in value) {
            fields[key] = value.val;
        } else {
            fields[key] = value;
        }
    });
    return fields;
}

/* Write the robot fields back in the canonical layout. */
function setRobotFields(fields) {
    const config = { ...(state.config || {}) };
    config.robot = fields;
    delete config.config;
    state.config = config;
    state.configEdited = true;
}

function renderRobotEditor() {
    const fields = robotFields();
    const box = el('robot-fields');
    box.innerHTML = '';
    ROBOT_FIELDS.forEach((spec) => {
        const raw = fields[spec.key];
        const label = document.createElement('label');
        label.textContent = spec.label;
        if (spec.unit) {
            const unit = document.createElement('span');
            unit.className = 'unit';
            unit.textContent = ` ${spec.unit}`;
            label.appendChild(unit);
        }
        const input = document.createElement('input');
        input.type = 'number';
        input.step = spec.step;
        if (spec.min !== undefined) input.min = spec.min;
        if (spec.max !== undefined) input.max = spec.max;
        input.dataset.key = spec.key;
        input.dataset.scale = spec.scale || 1;
        if (raw === undefined || raw === null || Number.isNaN(Number(raw))) {
            input.value = '';
        } else {
            input.value = Number(raw) * (spec.scale || 1);
        }
        input.addEventListener('input', () => {
            const value = Number(input.value);
            const invalid = input.value !== ''
                && (Number.isNaN(value)
                    || (spec.min !== undefined && value < spec.min)
                    || (spec.max !== undefined && value > spec.max));
            input.classList.toggle('invalid', invalid);
            if (invalid) return;
            const updated = robotFields();
            if (input.value === '') {
                delete updated[spec.key];
            } else {
                updated[spec.key] = value / (spec.scale || 1);
            }
            setRobotFields(updated);
            state.dirty = true;
            renderRobotFacts();
            updateMissionState();
            scheduleSessionSave();
        });
        box.appendChild(label);
        box.appendChild(input);
    });

    const ranges = el('robot-range-fields');
    ranges.innerHTML = '';
    const uncertainties = fields.uncertainties || {};
    RANGE_FIELDS.forEach((spec) => {
        const range = uncertainties[spec.key];
        [0, 1].forEach((end) => {
            const label = document.createElement('label');
            label.textContent = end === 0 ? `${spec.label} min` : `${spec.label} max`;
            if (spec.unit) {
                const unit = document.createElement('span');
                unit.className = 'unit';
                unit.textContent = ` ${spec.unit}`;
                label.appendChild(unit);
            }
            const input = document.createElement('input');
            input.type = 'number';
            input.step = 'any';
            input.dataset.key = spec.key;
            input.dataset.end = end;
            input.dataset.scale = spec.scale || 1;
            const value = Array.isArray(range) ? range[end] : undefined;
            input.value = (value === undefined || value === null) ? '' : Number(value) * (spec.scale || 1);
            input.addEventListener('input', () => {
                const updated = robotFields();
                const current = Array.isArray(updated.uncertainties?.[spec.key])
                    ? [...updated.uncertainties[spec.key]] : ['', ''];
                const number = input.value === '' ? '' : Number(input.value) / (spec.scale || 1);
                current[end] = number;
                if (current[0] !== '' && current[1] !== '' && Number(current[0]) > Number(current[1])) {
                    input.classList.add('invalid');
                    return;
                }
                input.classList.remove('invalid');
                updated.uncertainties = { ...(updated.uncertainties || {}), [spec.key]: current };
                setRobotFields(updated);
                state.dirty = true;
                scheduleSessionSave();
            });
            ranges.appendChild(label);
            ranges.appendChild(input);
        });
    });

    const note = el('robot-editor-note');
    const edited = Object.keys(state.config || {}).some((key) => key === 'robot');
    note.textContent = edited
        ? 'The studio sends the edited robot to the solver. Download it to keep a copy.'
        : 'Pick or upload a robot configuration to edit it.';
}

function downloadConfig() {
    if (!state.config) {
        toast('No robot configuration to download.', true);
        return;
    }
    const name = (state.configName || 'robot').replace(/[^a-z0-9_-]+/gi, '-');
    const blob = new Blob([JSON.stringify(state.config, null, 1)], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `${name}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    toast(`Downloaded ${name}.json`);
}

function renderPresets() {
    const box = el('presets');
    box.innerHTML = '';
    const order = ['fast', 'balanced', 'safe'];
    const descriptions = {
        fast: 'One candidate, no validation. Quick look at a path.',
        balanced: 'Two candidates, no validation.',
        safe: 'Three candidates, validated, controller export.',
    };
    order.forEach((name) => {
        const preset = state.presets[name];
        if (!preset) return;
        const label = document.createElement('label');
        label.className = 'preset';
        const input = document.createElement('input');
        input.type = 'radio';
        input.name = 'preset';
        input.value = name;
        input.addEventListener('change', () => applyPreset(name));
        const title = document.createElement('strong');
        title.textContent = name[0].toUpperCase() + name.slice(1);
        const detail = document.createElement('small');
        detail.textContent = descriptions[name] || '';
        label.appendChild(input);
        label.appendChild(title);
        label.appendChild(detail);
        box.appendChild(label);
    });
}

function applyPreset(name) {
    const preset = state.presets[name];
    if (!preset) return;
    el('opt-samples').value = preset.samples;
    el('opt-accuracy').value = preset.accuracy_weight;
    el('opt-candidates').value = preset.candidates;
    el('opt-validate').checked = Boolean(preset.validate);
    el('opt-export').value = preset.export_format;
    document.querySelectorAll('#presets input[type=radio]').forEach((radio) => {
        radio.checked = radio.value === name;
    });
    updateCandidatesHint();
    scheduleSessionSave();
}

function updateCandidatesHint() {
    const candidates = el('opt-candidates');
    const validate = el('opt-validate');
    candidates.disabled = !validate.checked;
    candidates.title = validate.checked
        ? 'Every candidate is solved and checked.'
        : 'Candidates are only compared when validation is on.';
}

/* ----------------------------------------------------------------- solve */

function collectOptions() {
    return {
        samples: Number(el('opt-samples').value),
        accuracy_weight: Number(el('opt-accuracy').value),
        candidates: Number(el('opt-candidates').value),
        timing: el('opt-timing').value,
        validate: el('opt-validate').checked,
        export_format: el('opt-export').value,
        controller_dt: Number(el('opt-controller-dt').value),
    };
}

async function solve() {
    if (state.waypoints.length < 2) {
        toast('Add at least two waypoints before solving.', true);
        return;
    }
    if (!state.config) {
        toast('Choose a robot configuration first.', true);
        return;
    }
    setBusy(true);
    setStatus('Solving...');
    el('result-status').textContent = 'Solving...';
    el('log').textContent = '';
    state.jobSummary = null;
    renderDownloads([]);
    try {
        const job = await api('/api/solve', {
            method: 'POST',
            body: JSON.stringify({
                name: el('mission-name').value,
                config: state.config,
                waypoints: state.waypoints.map(waypointToJson),
                options: collectOptions(),
            }),
        });
        state.jobId = job.id;
        pollJob();
    } catch (err) {
        setBusy(false);
        setStatus(err.message, 'error');
        el('result-status').innerHTML = `<span class="badge error">Failed</span> ${err.message}`;
        toast(err.message, true);
    }
}

function pollJob() {
    clearTimeout(state.pollTimer);
    state.pollTimer = setTimeout(async () => {
        try {
            const job = await api(`/api/jobs/${state.jobId}`);
            el('log').textContent = (job.log || []).join('\n');
            const report = parseValidationReport(job.log);
            if (report) {
                state.reportCandidate = null;
                renderValidationReport(report);
            }
            if (job.status === 'running') {
                setStatus('Solving...');
                pollJob();
                return;
            }
            setBusy(false);
            if (job.status === 'ok') {
                state.jobSummary = job.summary;
                state.trajectory = await api(`/api/jobs/${state.jobId}/trajectory`);
                state.trajectory = state.trajectory.trajectory.samples;
                playback.duration = state.trajectory.length
                    ? state.trajectory[state.trajectory.length - 1].t : 0;
                playback.time = 0;
                el('playback').value = 0;
                renderResult();
                renderDownloads(job.artifacts);
                const validated = el('opt-validate').checked;
                el('result-status').innerHTML = validated
                    ? '<span class="badge ok">Validated</span> trajectory written and checked.'
                    : '<span class="badge warn">Not validated</span> enable validation before driving.';
                setStatus('Done', 'ok');
            } else if (job.status === 'cancelled') {
                el('result-status').innerHTML = '<span class="badge warn">Cancelled</span>';
                setStatus('Cancelled', 'warn');
            } else {
                const reason = job.error || 'The solve failed.';
                el('result-status').innerHTML = `<span class="badge error">Failed</span> ${reason}`;
                setStatus('Failed', 'error');
                toast(reason, true);
            }
            draw();
        } catch (err) {
            setBusy(false);
            setStatus(err.message, 'error');
            toast(err.message, true);
        }
    }, 400);
}

function setBusy(busy) {
    el('btn-solve').disabled = busy;
    el('btn-cancel').hidden = !busy;
    el('btn-solve').textContent = busy ? 'Solving...' : 'Solve';
}

/* ---------------------------------------------------------------- canvas */

function resize() {
    const ratio = window.devicePixelRatio || 1;
    canvas.width = canvas.clientWidth * ratio;
    canvas.height = canvas.clientHeight * ratio;
    draw();
}

function screenToWorld(x, y) {
    return {
        x: (x - camera.offsetX) / camera.scale,
        y: -(y - camera.offsetY) / camera.scale,
    };
}

function worldToScreen(x, y) {
    return {
        x: x * camera.scale + camera.offsetX,
        y: -y * camera.scale + camera.offsetY,
    };
}

function frameAll() {
    const points = state.waypoints.slice();
    if (state.trajectory) {
        state.trajectory.forEach((sample) => points.push(sample));
    }
    if (!points.length) {
        camera.offsetX = canvas.width / 2;
        camera.offsetY = canvas.height / 2;
        camera.scale = 400;
        return;
    }
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const pad = 0.25;
    const width = Math.max(maxX - minX + 2 * pad, 0.5);
    const height = Math.max(maxY - minY + 2 * pad, 0.5);
    const ratio = canvas.width / canvas.height;
    camera.scale = Math.min(canvas.width / width, canvas.height / (height * ratio));
    camera.scale = Math.max(Math.min(camera.scale, 4000), 20);
    const center = worldToScreen((minX + maxX) / 2, (minY + maxY) / 2);
    camera.offsetX = canvas.width / 2 - (center.x - camera.offsetX);
    camera.offsetY = canvas.height / 2 - (center.y - camera.offsetY);
}

function draw() {
    if (!ctx) return;
    const ratio = window.devicePixelRatio || 1;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawGrid();

    if (state.showPath && state.waypoints.length > 1) {
        // Segment by segment, so a segment that failed validation stands out.
        ctx.save();
        ctx.setLineDash([6 * ratio, 6 * ratio]);
        ctx.lineWidth = 2 * ratio;
        for (let i = 0; i < state.waypoints.length - 1; i++) {
            const failing = state.failingSegments.has(i);
            const worst = state.worstSegment === i;
            if (failing) ctx.strokeStyle = 'rgba(248, 113, 113, 0.95)';
            else if (worst) ctx.strokeStyle = 'rgba(251, 191, 36, 0.9)';
            else ctx.strokeStyle = 'rgba(148, 163, 184, 0.7)';
            const a = worldToScreen(state.waypoints[i].x, state.waypoints[i].y);
            const b = worldToScreen(state.waypoints[i + 1].x, state.waypoints[i + 1].y);
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
        }
        ctx.restore();
    }

    if (state.trajectory) {
        drawEnvelope(state.trajectory);
        drawPath(state.trajectory, 'rgba(56, 189, 248, 0.25)', 10 * ratio);
        drawPath(state.trajectory, '#38bdf8', 2.5 * ratio);
    }

    drawWaypoints();
    drawRobot();
    updatePlaybackLabel();
}

function drawGrid() {
    const ratio = window.devicePixelRatio || 1;
    const step = 0.25 * camera.scale;
    if (step < 8) return;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = camera.offsetX % step; x < canvas.width; x += step) {
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
    }
    for (let y = camera.offsetY % step; y < canvas.height; y += step) {
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
    }
    ctx.stroke();
}

function drawPath(samples, color, width) {
    if (!samples || samples.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    samples.forEach((sample, index) => {
        const p = worldToScreen(sample.x, sample.y);
        if (index === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();
}

function drawEnvelope(samples) {
    if (!state.showEnvelope) return;
    const withEnv = samples.filter((s) => s.reachability_envelope);
    if (withEnv.length < 2) return;
    const ratio = window.devicePixelRatio || 1;
    ctx.fillStyle = 'rgba(52, 211, 153, 0.12)';
    ctx.strokeStyle = 'rgba(52, 211, 153, 0.45)';
    ctx.lineWidth = 1 * ratio;
    ctx.setLineDash([4 * ratio, 4 * ratio]);
    ctx.beginPath();
    withEnv.forEach((sample, index) => {
        const p = worldToScreen(sample.reachability_envelope.x_max, sample.reachability_envelope.y_max);
        if (index === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
    });
    for (let i = withEnv.length - 1; i >= 0; i--) {
        const sample = withEnv[i];
        const p = worldToScreen(sample.reachability_envelope.x_min, sample.reachability_envelope.y_min);
        ctx.lineTo(p.x, p.y);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.setLineDash([]);
}

function drawWaypoints() {
    const ratio = window.devicePixelRatio || 1;
    state.waypoints.forEach((point, index) => {
        const p = worldToScreen(point.x, point.y);
        const active = index === state.selected;
        const failing = state.failingSegments.has(index) || state.failingSegments.has(index - 1);
        ctx.fillStyle = active ? '#38bdf8' : (point.heading === null ? '#94a3b8' : '#f472b6');
        ctx.beginPath();
        ctx.arc(p.x, p.y, (active ? 7 : 5.5) * ratio, 0, Math.PI * 2);
        ctx.fill();

        if (failing) {
            ctx.strokeStyle = '#f87171';
            ctx.lineWidth = 2 * ratio;
            ctx.beginPath();
            ctx.arc(p.x, p.y, 11 * ratio, 0, Math.PI * 2);
            ctx.stroke();
        }

        if (point.heading !== null) {
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2 * ratio;
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(p.x + Math.cos(point.heading) * 22 * ratio,
                p.y - Math.sin(point.heading) * 22 * ratio);
            ctx.stroke();
        }

        ctx.fillStyle = '#94a3b8';
        ctx.font = `${11 * ratio}px system-ui, sans-serif`;
        ctx.fillText(`W${index}`, p.x + 10 * ratio, p.y - 8 * ratio);
    });
}

function drawRobot() {
    if (!state.trajectory || !state.trajectory.length) return;
    const samples = state.trajectory;
    let sample = samples[0];
    for (let i = 0; i < samples.length - 1; i++) {
        if (samples[i + 1].t >= playback.time) {
            const span = samples[i + 1].t - samples[i].t;
            const ratio = span > 1e-9 ? (playback.time - samples[i].t) / span : 0;
            sample = {
                x: samples[i].x + (samples[i + 1].x - samples[i].x) * ratio,
                y: samples[i].y + (samples[i + 1].y - samples[i].y) * ratio,
                heading: samples[i].heading
                    + (samples[i + 1].heading - samples[i].heading) * ratio,
            };
            break;
        }
        sample = samples[samples.length - 1];
    }
    const ratio = window.devicePixelRatio || 1;
    const p = worldToScreen(sample.x, sample.y);
    const robot = (state.config && (state.config.robot || state.config)) || {};
    const track = robot.track_width || 0.0965;
    const size = track * camera.scale;

    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(-sample.heading);
    ctx.fillStyle = 'rgba(56, 189, 248, 0.25)';
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2 * ratio;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(-size / 2, -size / 2, size, size, 4 * ratio);
    else ctx.rect(-size / 2, -size / 2, size, size);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
}

function updatePlaybackLabel() {
    el('playback-time').textContent = `${playback.time.toFixed(2)}s / ${playback.duration.toFixed(2)}s`;
}

function hitTest(sx, sy) {
    const ratio = window.devicePixelRatio || 1;
    for (let i = state.waypoints.length - 1; i >= 0; i--) {
        const p = worldToScreen(state.waypoints[i].x, state.waypoints[i].y);
        if (Math.hypot(p.x - sx, p.y - sy) <= 12 * ratio) return i;
    }
    return -1;
}

/* --------------------------------------------------------------- events */

let drag = null;

canvas.addEventListener('pointerdown', (event) => {
    const ratio = window.devicePixelRatio || 1;
    const sx = event.offsetX * ratio;
    const sy = event.offsetY * ratio;
    const index = hitTest(sx, sy);
    if (index >= 0) {
        state.selected = index;
        drag = { kind: 'waypoint', index, moved: false };
        renderWaypointList();
        renderEditor();
    } else {
        drag = { kind: 'pan', sx, sy, offsetX: camera.offsetX, offsetY: camera.offsetY, moved: false };
    }
    try {
        canvas.setPointerCapture(event.pointerId);
    } catch (err) {
        // Synthetic or already released pointers cannot be captured.
    }
});

canvas.addEventListener('pointermove', (event) => {
    const ratio = window.devicePixelRatio || 1;
    if (!drag) return;
    const sx = event.offsetX * ratio;
    const sy = event.offsetY * ratio;
    if (drag.kind === 'pan') {
        camera.offsetX = drag.offsetX + (sx - drag.sx);
        camera.offsetY = drag.offsetY + (sy - drag.sy);
        if (Math.hypot(sx - drag.sx, sy - drag.sy) > 3) drag.moved = true;
    } else {
        const world = screenToWorld(sx, sy);
        const point = state.waypoints[drag.index];
        if (point) {
            point.x = Math.round(world.x * 1000) / 1000;
            point.y = Math.round(world.y * 1000) / 1000;
            drag.moved = true;
            state.dirty = true;
            renderEditor();
        }
    }
    draw();
});

canvas.addEventListener('pointerup', (event) => {
    const ratio = window.devicePixelRatio || 1;
    if (drag && drag.kind === 'pan' && !drag.moved) {
        const world = screenToWorld(event.offsetX * ratio, event.offsetY * ratio);
        if (state.waypoints.length === 0) {
            state.waypoints.push(blankWaypoint(0, 0, 0));
        } else {
            state.waypoints.push(blankWaypoint(
                Math.round(world.x * 1000) / 1000,
                Math.round(world.y * 1000) / 1000,
                null));
        }
        state.selected = state.waypoints.length - 1;
        state.dirty = true;
        renderAll();
        pushHistory();
    } else if (drag && drag.kind === 'waypoint' && drag.moved) {
        // One history entry per drag, not one per mouse move.
        pushHistory();
    }
    drag = null;
    draw();
});

canvas.addEventListener('wheel', (event) => {
    event.preventDefault();
    const ratio = window.devicePixelRatio || 1;
    const sx = event.offsetX * ratio;
    const sy = event.offsetY * ratio;
    const before = screenToWorld(sx, sy);
    const factor = Math.exp(-event.deltaY * 0.0015);
    camera.scale = Math.max(Math.min(camera.scale * factor, 4000), 20);
    const after = screenToWorld(sx, sy);
    camera.offsetX += (after.x - before.x) * camera.scale;
    camera.offsetY -= (after.y - before.y) * camera.scale;
    draw();
}, { passive: false });

document.addEventListener('keydown', (event) => {
    const modifier = event.ctrlKey || event.metaKey;
    if (modifier && event.key.toLowerCase() === 'z') {
        event.preventDefault();
        if (event.shiftKey) redo();
        else undo();
        return;
    }
    if (modifier && event.key.toLowerCase() === 'y') {
        event.preventDefault();
        redo();
        return;
    }
    if (modifier && event.key.toLowerCase() === 's') {
        event.preventDefault();
        saveMission();
        return;
    }
    const tag = (event.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'select' || tag === 'textarea') return;
    if (event.key === ' ') {
        event.preventDefault();
        el('btn-play').click();
        return;
    }
    if ((event.key === 'Delete' || event.key === 'Backspace') && state.selected >= 0) {
        event.preventDefault();
        deleteSelected();
    }
});

function deleteSelected() {
    if (state.selected < 0) return;
    state.waypoints.splice(state.selected, 1);
    state.selected = Math.min(state.selected, state.waypoints.length - 1);
    state.dirty = true;
    renderAll();
    pushHistory();
}

function updateField(field, value) {
    const point = currentWaypoint();
    if (!point) return;
    point[field] = value;
    state.dirty = true;
    renderWaypointList();
    updateMissionState();
}

el('wp-x').addEventListener('input', (e) => updateField('x', Number(e.target.value)));
el('wp-y').addEventListener('input', (e) => updateField('y', Number(e.target.value)));
el('wp-heading').addEventListener('input', (e) => {
    const point = currentWaypoint();
    if (!point) return;
    const degrees = Number(e.target.value);
    point.heading = Number.isFinite(degrees) ? degrees * Math.PI / 180 : null;
    state.dirty = true;
    renderWaypointList();
    updateMissionState();
    draw();
});
el('wp-free-heading').addEventListener('change', (e) => {
    const point = currentWaypoint();
    if (!point) return;
    point.heading = e.target.checked ? null : 0;
    state.dirty = true;
    renderEditor();
    renderWaypointList();
    updateMissionState();
    draw();
    pushHistory();
});
el('wp-stop').addEventListener('change', (e) => {
    updateField('stop', e.target.checked);
    pushHistory();
});
el('wp-event').addEventListener('input', (e) => updateField('event', e.target.value.trim()));
el('wp-event').addEventListener('change', () => pushHistory());
el('wp-precision').addEventListener('change', (e) => {
    updateField('precision_mode', e.target.checked);
    pushHistory();
});
el('wp-threshold').addEventListener('input', (e) => {
    const value = e.target.value === '' ? null : Number(e.target.value) / 100;
    updateField('error_threshold', value);
});
el('wp-threshold').addEventListener('change', () => pushHistory());
el('wp-x').addEventListener('change', () => pushHistory());
el('wp-y').addEventListener('change', () => pushHistory());
el('wp-heading').addEventListener('change', () => pushHistory());
el('btn-delete').addEventListener('click', deleteSelected);
el('btn-move-up').addEventListener('click', () => moveSelected(-1));
el('btn-move-down').addEventListener('click', () => moveSelected(1));

function moveSelected(delta) {
    const index = state.selected;
    const target = index + delta;
    if (index < 0 || target < 0 || target >= state.waypoints.length) return;
    const [point] = state.waypoints.splice(index, 1);
    state.waypoints.splice(target, 0, point);
    state.selected = target;
    state.dirty = true;
    renderAll();
    pushHistory();
}

el('btn-solve').addEventListener('click', solve);
el('btn-cancel').addEventListener('click', async () => {
    if (!state.jobId) return;
    try {
        await api(`/api/jobs/${state.jobId}/cancel`, { method: 'POST' });
    } catch (err) {
        toast(err.message, true);
    }
});

el('btn-frame').addEventListener('click', () => { frameAll(); draw(); });
el('toggle-envelope').addEventListener('change', (e) => {
    state.showEnvelope = e.target.checked;
    draw();
});
el('toggle-path').addEventListener('change', (e) => {
    state.showPath = e.target.checked;
    draw();
});

el('btn-play').addEventListener('click', () => {
    playback.playing = !playback.playing;
    el('btn-play').textContent = playback.playing ? 'Pause' : 'Play';
});
el('playback').addEventListener('input', (e) => {
    playback.time = (Number(e.target.value) / 1000) * playback.duration;
    playback.playing = false;
    el('btn-play').textContent = 'Play';
    draw();
});

el('btn-new').addEventListener('click', () => {
    setWaypoints([blankWaypoint(0, 0, 0)]);
    el('mission-name').value = 'mission';
    frameAll();
    draw();
});

el('btn-undo').addEventListener('click', undo);
el('btn-redo').addEventListener('click', redo);
el('btn-download').addEventListener('click', downloadMission);

el('btn-save').addEventListener('click', saveMission);

async function saveMission() {
    if (state.waypoints.length < 2) {
        toast('A mission needs at least two waypoints.', true);
        return;
    }
    try {
        const result = await api('/api/missions', {
            method: 'POST',
            body: JSON.stringify({
                name: el('mission-name').value,
                waypoints: state.waypoints.map(waypointToJson),
            }),
        });
        state.dirty = false;
        state.savedName = result.name;
        updateMissionState();
        toast(`Saved missions/${result.name}.json`);
        await loadExamples();
    } catch (err) {
        toast(err.message, true);
    }
}

el('mission-list').addEventListener('change', (e) => {
    const mission = state.missions.find((m) => m.name === e.target.value);
    if (!mission) return;
    // The previous mission stays in the undo history, so nothing is lost.
    setWaypoints(mission.data);
    el('mission-name').value = mission.name;
    frameAll();
    draw();
    scheduleSessionSave();
});

el('config-list').addEventListener('change', (e) => {
    const config = state.configs.find((c) => c.name === e.target.value);
    if (!config) return;
    state.config = config.data;
    state.configName = config.name;
    state.configEdited = false;
    renderRobotFacts();
    renderRobotEditor();
    scheduleSessionSave();
});

el('btn-download-config').addEventListener('click', downloadConfig);

el('btn-upload-mission').addEventListener('click', () => el('file-mission').click());
el('file-mission').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
        const data = JSON.parse(await file.text());
        if (!Array.isArray(data)) throw new Error('A mission file must contain a list of waypoints.');
        setWaypoints(data);
        el('mission-name').value = file.name.replace(/\.json$/i, '');
        frameAll();
        draw();
    } catch (err) {
        toast(err.message, true);
    }
    e.target.value = '';
});

el('btn-upload-config').addEventListener('click', () => el('file-config').click());
el('file-config').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
        const data = JSON.parse(await file.text());
        state.config = data;
        state.configName = file.name.replace(/\.json$/i, '');
        state.configEdited = false;
        renderRobotFacts();
        renderRobotEditor();
        scheduleSessionSave();
    } catch (err) {
        toast(err.message, true);
    }
    e.target.value = '';
});

el('opt-validate').addEventListener('change', () => {
    updateCandidatesHint();
    scheduleSessionSave();
});

['opt-samples', 'opt-accuracy', 'opt-candidates', 'opt-timing', 'opt-export', 'opt-controller-dt']
    .forEach((id) => el(id).addEventListener('change', () => {
        updateCandidatesHint();
        syncPresetRadio();
        scheduleSessionSave();
    }));

el('mission-name').addEventListener('input', scheduleSessionSave);

window.addEventListener('resize', resize);

/* ------------------------------------------------------------------ loop */

function animate() {
    if (playback.playing && playback.duration > 0) {
        playback.time += 0.02;
        if (playback.time > playback.duration) playback.time = 0;
        el('playback').value = Math.round((playback.time / playback.duration) * 1000);
    }
    draw();
    requestAnimationFrame(animate);
}

/* ------------------------------------------------------------------ init */

async function loadExamples() {
    const data = await api('/api/examples');
    state.configs = data.configs;
    state.missions = data.missions;

    const configList = el('config-list');
    configList.innerHTML = '';
    data.configs.forEach((config) => {
        const option = document.createElement('option');
        option.value = config.name;
        option.textContent = config.name;
        configList.appendChild(option);
    });
    if (state.configName && !data.configs.some((c) => c.name === state.configName)) {
        // A config that is not one of the examples (uploaded, or restored).
        const option = document.createElement('option');
        option.value = state.configName;
        option.textContent = `${state.configName} (custom)`;
        configList.appendChild(option);
    }
    if (state.configName) configList.value = state.configName;

    const missionList = el('mission-list');
    missionList.innerHTML = '';
    const blank = document.createElement('option');
    blank.value = '';
    blank.textContent = 'Choose a mission...';
    missionList.appendChild(blank);
    data.missions.forEach((mission) => {
        const option = document.createElement('option');
        option.value = mission.name;
        option.textContent = mission.source === 'saved' ? `${mission.name} (saved)` : mission.name;
        missionList.appendChild(option);
    });
}

async function init() {
    try {
        const health = await api('/api/health');
        state.health = health;
        state.presets = health.presets || {};
        renderPresets();

        const initial = await api('/api/initial');
        if (initial.config) {
            state.config = initial.config;
            state.configName = initial.config_name || '';
        }

        // A mission left open in this browser comes back automatically.
        const session = loadSession();
        const restored = Boolean(session && Array.isArray(session.waypoints) && session.waypoints.length);
        if (restored) {
            setWaypoints(session.waypoints);
            el('mission-name').value = session.name || 'mission';
            if (session.config) {
                state.config = session.config;
                state.configName = session.configName || '';
                state.configEdited = Boolean(session.configEdited);
            }
            applyOptionsToForm(session.options);
        } else if (initial.waypoints) {
            setWaypoints(initial.waypoints);
            el('mission-name').value = state.configName || 'mission';
            applyPreset('balanced');
        } else {
            setWaypoints([blankWaypoint(0, 0, 0)]);
            applyPreset('balanced');
        }
        resetHistory();

        await loadExamples();
        renderRobotFacts();
        renderRobotEditor();
        updateCandidatesHint();

        camera.offsetX = canvas.width / 2;
        camera.offsetY = canvas.height / 2;
        resize();
        frameAll();
        draw();
        setStatus('Ready', 'ok');
        animate();
        if (restored) {
            toast(`Restored your last mission (${state.waypoints.length} waypoints).`);
        }
    } catch (err) {
        setStatus(err.message, 'error');
        toast(err.message, true);
    }
}

init();
