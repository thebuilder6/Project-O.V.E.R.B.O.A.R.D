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
    showEnvelopeBoxes: false,
    showPath: true,
    jobId: null,
    // One timer per poll loop. They used to share one, so starting any
    // second loop cancelled the first: the solve's poll stopped firing and
    // the studio looked hung although the server had finished.
    pollTimer: null,
    pushPollTimer: null,
    collectPollTimer: null,
    health: null,
    presets: {},
    /* The mission on screen has moved on since the last solve: the
       trajectory that is on the field, if there is one, is not the answer
       to the mission in front of the operator any more. Every edit sets
       it, and a completed solve clears it when it solved the mission that
       is on the screen. */
    dirty: false,
    savedName: '',
    /* What is on disk, so the studio can say whether the mission on the
       screen is still the one that was saved. */
    savedFingerprint: null,
    /* The fingerprint of the mission being solved and of the one that was
       solved. Two moments with the same fingerprint are the same mission,
       so a trajectory can be checked against the screen. */
    solvingFingerprint: null,
    configEdited: false,
    jobWallS: null,
    /* The segments the validator found failing, and the one it found
       worst. A Set and a number, so the field can draw them. */
    failingSegments: new Set(),
    worstSegment: -1,
    reportCandidate: null,
    /* The verdict of the last run. Null means there is none: a solve
       without validation has nothing to show. */
    validationReport: null,
    // One retry budget per poll loop: they used to share one, so a solve
    // that had used its three retries left the push with none.
    pollRetries: 0,
    pushPollRetries: 0,
    solveJobId: null,
    solveHasTrajectory: false,
    /* Whether the last solve passed the validator. The push to the robot
       needs it: a trajectory that was never checked is not an approved one. */
    solveValidated: false,
    /* The score's decision output: not a probability of hardware success. */
    hubApproved: false,
    runOutcome: 'unvalidated',
    jobScore: null,
    /* Measured lever tries for this mission session; ranked only after a run. */
    leverHistory: [],
    pendingLever: null,
    missionScoring: {},
    pushJobId: null,
    measuredPoses: [],
    robotInfo: { events: [], geometry: {} },
    jobResult: null,
    solveHistory: [],
    ghostStates: null,
    plot: 'speed',
    hubConn: 'ble',
    hubName: '',
    /* The mat under the mission. FIRST's artwork is not ours to ship, so
       the image is the user's; the size is ours to know. */
    mat: { image: null, name: '', widthM: 2.004, heightM: 1.143 },
    showMat: true,
    /* The earlier solves, so a new one can be compared with them. */
    runs: [],
    showRuns: true,
    /* The solver-side overlays: the trajectory coloured by actuator
       saturation, and the late-apex cues on the curved segments. */
    showSaturation: true,
    showApex: true,
    /* Every solved candidate's path, so the three frontier plans are on the
       field together and can be compared (the CLI writes them for the run).
       The candidate the chips point at is drawn stronger. */
    candidatePaths: [],
    /* The candidate the operator chose for the next solve. It is consumed by
       that solve, so a later run is not quietly pinned to the same seed. */
    pickCandidate: null,
    /* What the pointer is over: a mission model or a waypoint. */
    hoverObstacle: null,
    hoverWaypoint: -1,
    /* The field: the mission models standing on the mat, and the collisions
       the last solve had with them. */
    fields: {},
    seasons: [],
    season: '',
    collisions: [],
    /* The band around each mission model: the field's placement uncertainty.
       It is drawn until the operator clicks it off in the key. */
    showPlacement: true,
};

/* Undo/redo history: a stack of mission snapshots. */
const history = { stack: [], index: -1, limit: 100 };

/* The mission is kept in the browser so a reload never loses the work. */
const SESSION_KEY = 'overboard.studio.session.v1';
// The canvas overlays the legend switches; they ride in the session too, so a
// reload does not quietly switch the mat or the envelope back on.
const OVERLAY_FLAGS = ['showMat', 'showEnvelope', 'showEnvelopeBoxes', 'showPath', 'showRuns',
    'showSaturation', 'showApex', 'showPlacement'];
let sessionTimer = null;

/* Camera: pixels per metre plus a device-pixel offset. */
const camera = { scale: 400, offsetX: 0, offsetY: 0 };
const playback = { time: 0, playing: false, duration: 0 };

/* ------------------------------------------------------------------ tabs

   Each side column is a strip of tabs over one panel. The markup declares
   the tabs and the panels; this wires them: the ARIA state, the roving
   focus and the arrow keys. Adding a tab is a button and a panel, and
   nothing else in this file has to know about it. */
const tabStrips = [];

function setupTabs() {
    document.querySelectorAll('[role="tablist"]').forEach((strip) => {
        const buttons = Array.from(strip.querySelectorAll('[role="tab"]'));
        if (!buttons.length) return;
        const entry = { strip, buttons, panels: {} };
        buttons.forEach((button) => {
            const panelId = button.getAttribute('aria-controls');
            entry.panels[panelId] = el(panelId);
            button.addEventListener('click', () => showTab(strip, panelId));
            button.addEventListener('keydown', (event) => {
                const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
                if (!step && event.key !== 'Home' && event.key !== 'End') return;
                event.preventDefault();
                const current = buttons.indexOf(button);
                let next = current;
                if (event.key === 'Home') next = 0;
                else if (event.key === 'End') next = buttons.length - 1;
                else next = (current + step + buttons.length) % buttons.length;
                buttons[next].focus();
                showTab(strip, buttons[next].getAttribute('aria-controls'));
            });
        });
        tabStrips.push(entry);
        const first = buttons.find((button) => button.getAttribute('aria-selected') === 'true')
            || buttons[0];
        showTab(strip, first.getAttribute('aria-controls'));
    });
}

/* Show one panel of a strip, and hide the rest. The scroll position of the
   column belongs to the panel that was in it, so each panel keeps its own. */
function showTab(strip, panelId) {
    const entry = tabStrips.find((item) => item.strip === strip);
    if (!entry) return;
    const scroller = entry.strip.closest('.panel');
    const current = entry.buttons.find((button) => button.getAttribute('aria-selected') === 'true');
    const currentId = current ? current.getAttribute('aria-controls') : null;
    if (scroller && currentId && currentId !== panelId) {
        const leaving = entry.panels[currentId];
        if (leaving) leaving._scroll = scroller.scrollTop;
    }
    entry.buttons.forEach((button) => {
        const selected = button.getAttribute('aria-controls') === panelId;
        button.setAttribute('aria-selected', selected ? 'true' : 'false');
        button.tabIndex = selected ? 0 : -1;
    });
    Object.entries(entry.panels).forEach(([id, panel]) => {
        if (panel) panel.hidden = id !== panelId;
    });
    if (scroller) {
        const panel = entry.panels[panelId];
        scroller.scrollTop = panel && panel._scroll !== undefined ? panel._scroll : 0;
    }
    // A hidden panel has no size, so anything drawn in it has to be drawn
    // again now that it is on screen.
    if (panelId === 'panel-plots') drawPlot();
}

/* Show a panel by its id, whichever strip it belongs to. */
function showPanel(panelId) {
    const entry = tabStrips.find((item) => item.panels[panelId]);
    if (!entry) return false;
    showTab(entry.strip, panelId);
    return true;
}

/* ------------------------------------------------------------ the plots

   The plots the studio can draw. The tab strip and the drawing both come
   from this list, so a new plot is one entry and nothing else changes. */
const PLOTS = [
    { key: 'speed', label: 'Speed', build: buildSpeedPlot },
    { key: 'force', label: 'Force', build: buildForcePlot },
    { key: 'accel', label: 'Acceleration', build: buildAccelPlot },
    { key: 'turn', label: 'Turn rate', build: buildTurnPlot },
    { key: 'heading', label: 'Heading', build: buildHeadingPlot },
    { key: 'envelope', label: 'Envelope', build: buildEnvelopePlot },
    { key: 'convergence', label: 'Convergence', build: buildConvergencePlot },
];

/* The key to the field's colours, and the one home of the overlay switches.
   The legend is built from this list, so a new drawn thing has one entry here
   and the key cannot go stale. The key is live: an entry for something that
   is not on the field is hidden, so the list is a key to what the operator
   can see, not a list of possibilities. A `toggle` on an entry makes it the
   switch for that overlay: the key is where the operator looks to see what
   is drawn, so the switch belongs there and nowhere else. */
const CANVAS_LEGEND = [
    { key: 'mat', color: 'rgba(148, 163, 184, 0.35)', label: 'Mat picture', toggle: 'showMat' },
    { key: 'trajectory', color: '#38bdf8', label: 'Planned trajectory' },
    { key: 'candidates', color: '#a78bfa', label: 'Compared candidates', dash: true },
    { key: 'envelope', color: 'rgba(52, 211, 153, 0.65)', label: 'Reachability envelope', dash: true, toggle: 'showEnvelope' },
    { key: 'envelopeBoxes', color: 'rgba(52, 211, 153, 0.45)', label: 'Envelope as its sample boxes', toggle: 'showEnvelopeBoxes' },
    { key: 'saturation', color: 'rgba(251, 191, 36, 0.95)', label: 'At the actuator limit', toggle: 'showSaturation' },
    { key: 'slipping', color: 'rgba(248, 113, 113, 0.95)', label: 'Over the actuator limit', toggle: 'showSaturation' },
    { key: 'apex', color: '#f59e0b', label: 'Late-apex target', dot: true, toggle: 'showApex' },
    { key: 'bisector', color: 'rgba(226, 232, 240, 0.8)', label: 'Momentum bisector', dash: true, toggle: 'showApex' },
    { key: 'path', color: 'rgba(148, 163, 184, 0.85)', label: 'Path between waypoints', dash: true, toggle: 'showPath' },
    { key: 'ghost', color: 'rgba(226, 232, 240, 0.6)', label: "Solver's first guess", dash: true },
    { key: 'runs', color: 'rgba(148, 163, 184, 0.5)', label: 'Earlier solves', toggle: 'showRuns' },
    { key: 'heading', color: '#f472b6', label: 'Waypoint with a heading', dot: true },
    { key: 'free', color: '#94a3b8', label: 'Waypoint, heading free', dot: true },
    { key: 'measured', color: '#34d399', label: 'Measured pose from the robot', dot: true },
    { key: 'models', color: 'rgba(56, 189, 248, 0.95)', label: 'Mission model on the mat' },
    { key: 'placement', color: 'rgba(251, 191, 36, 0.7)', label: 'Placement uncertainty', dash: true, toggle: 'showPlacement' },
    { key: 'collision', color: 'rgba(248, 113, 113, 0.95)', label: 'Collision with a mission model' },
    { key: 'failing', color: 'rgba(251, 191, 36, 0.95)', label: 'Failing segment, uncertain collision' },
];

/* ------------------------------------------------------------------ api */

/* The field the mission is drawn on. The server owns the geometry and the
   collision maths; the browser draws what it is told. */
async function loadFields() {
    try {
        const data = await api('/api/fields');
        state.fields = data.fields || {};
        state.seasons = data.seasons || [];
    } catch (err) {
        state.fields = {};
        state.seasons = [];
    }
    const select = el('field-season');
    if (!select) return;
    select.innerHTML = '';
    if (!state.seasons.length) {
        const option = document.createElement('option');
        option.textContent = 'no field data';
        option.value = '';
        select.appendChild(option);
        select.disabled = true;
        return;
    }
    select.disabled = false;
    state.seasons.forEach((season) => {
        const option = document.createElement('option');
        option.value = season.id;
        option.textContent = season.label || season.id;
        select.appendChild(option);
    });
    if (!state.season || !state.fields[state.season]) {
        state.season = state.seasons[0].id;
    }
    select.value = state.season;
}

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
    // The Solve panel says the same thing, so filling it in is not blind.
    const mirror = el('solve-status');
    if (mirror) mirror.textContent = text;
}

/* The result line: a badge and a sentence. The sentence comes from the
   server, so it is put in as text and never as HTML. */
function setResultStatus(badgeKind, badgeText, message) {
    const node = el('result-status');
    node.textContent = '';
    const badge = document.createElement('span');
    badge.className = `badge ${badgeKind}`;
    badge.textContent = badgeText;
    node.appendChild(badge);
    if (message) node.appendChild(document.createTextNode(` ${message}`));
}

/* The one line in the header that says where the last solve stands: the
   time, the path and the envelope. It is always on screen, so the answer
   to "how did it go" is never somewhere else. */
function resultHeadline() {
    const summary = state.jobSummary;
    if (!summary) return '';
    const parts = [];
    if (Number.isFinite(summary.duration_s)) parts.push(`${summary.duration_s.toFixed(2)} s`);
    if (Number.isFinite(summary.path_length_m)) parts.push(`${summary.path_length_m.toFixed(2)} m`);
    if (summary.has_envelope && Number.isFinite(summary.max_envelope_m)) {
        parts.push(`envelope ${(summary.max_envelope_m * 100).toFixed(1)} cm`);
    }
    return parts.join(' | ');
}

/* -------------------------------------------------------------- mission */

function blankWaypoint(x = 0, y = 0, heading = 0) {
    return { x, y, heading, stop: false, event: '', precision_mode: false, error_threshold: null };
}

/* The mission on screen changed. The trajectory that is on the field, if
   there is one, is not the answer to it any more, so the push button has
   to follow. */
function markMissionChanged() {
    state.dirty = true;
    state.leverHistory = [];
    renderLeverHistory();
    updatePushButton();
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
    /* Another mission is on the screen now, so the trajectory that is on
       the field, if there is one, answers a different mission. */
    markMissionChanged();
    state.savedName = '';
    state.savedFingerprint = null;
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

/* Two names for the same mission. The server slugs the name it is given
   (lower case, and anything that is not a word character becomes a
   dash), so the two are compared without their punctuation. */
function sameMissionName(one, other) {
    const plain = (text) => String(text || '').trim().toLowerCase()
        .replace(/[^a-z0-9]+/g, '');
    const left = plain(one);
    const right = plain(other);
    if (!left || !right) return true;
    return left === right;
}

function currentWaypoint() {
    if (state.selected < 0 || state.selected >= state.waypoints.length) return null;
    return state.waypoints[state.selected];
}

/* What the studio is showing, as one string. Two moments with the same
   fingerprint are the same thing, so a trajectory can be checked against
   the screen. The robot is in it, because the trajectory depends on it;
   the mission's name is not, because a name does not change the path.
   Without the robot it is the mission file's own content, which is what
   the saved state is compared against. */
function missionFingerprint(withRobot = true) {
    const parts = { waypoints: state.waypoints.map(waypointToJson) };
    if (withRobot) {
        const robot = robotFields();
        const fields = {};
        Object.keys(robot).sort().forEach((key) => { fields[key] = robot[key]; });
        parts.robot = fields;
    }
    return JSON.stringify(parts);
}

/* --------------------------------------------------------- history */

function snapshot() {
    return {
        waypoints: JSON.parse(JSON.stringify(state.waypoints)),
        selected: state.selected,
        robot: JSON.parse(JSON.stringify(robotFields())),
        configName: state.configName,
    };
}

/* Record the current mission so it can be brought back. The robot config is
 * part of the snapshot, so a config edit can be undone like a waypoint edit. */
function pushHistory() {
    const current = history.stack[history.index];
    const now = snapshot();
    if (current && JSON.stringify(current.waypoints) === JSON.stringify(now.waypoints)
        && JSON.stringify(current.robot) === JSON.stringify(now.robot)) {
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
    if (snap.robot) {
        setRobotFields(JSON.parse(JSON.stringify(snap.robot)));
        state.configName = snap.configName || state.configName;
        renderRobotFacts();
        renderRobotEditor();
        renderRobotChecks();
    }
    markMissionChanged();
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
            scoring: state.missionScoring,
            options: collectOptions(),
            hubConn: state.hubConn,
            hubName: state.hubName,
            model: state.model,
            useModel: state.useModel,
            matWidthM: state.mat.widthM,
            matHeightM: state.mat.heightM,
            season: state.season,
            overlays: Object.fromEntries(OVERLAY_FLAGS.map((key) => [key, state[key]])),
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
    if (options.auto_mesh !== undefined) el('opt-auto-mesh').checked = Boolean(options.auto_mesh);
    if (options.accuracy_weight !== undefined) el('opt-accuracy').value = options.accuracy_weight;
    if (options.candidates !== undefined) el('opt-candidates').value = options.candidates;
    if (options.force_weight !== undefined) el('opt-force-weight').value = options.force_weight;
    if (options.propagation !== undefined) {
        el('opt-propagation').value = options.propagation === 'tick' ? 'tick' : 'plant';
    }
    if (options.validate !== undefined) el('opt-validate').checked = options.validate;
    if (options.envelope_limit !== undefined) {
        el('opt-envelope-limit').value = Number(options.envelope_limit) * 100;
    }
    if (options.envelope_limit_per_metre !== undefined) {
        el('opt-envelope-limit-per-metre').value = Number(options.envelope_limit_per_metre) * 100;
    }
    if (options.best_effort !== undefined) {
        el('opt-best-effort').checked = Boolean(options.best_effort);
    }
    if (options.fast_polish !== undefined) el('opt-fast-polish').checked = options.fast_polish;
    if (options.target_time !== undefined) el('opt-target-time').value = options.target_time;
    if (options.max_time !== undefined) el('opt-max-time').value = options.max_time;
    if (options.jerk_target !== undefined) el('opt-jerk-target').value = options.jerk_target;
    if (options.jerk_max !== undefined) el('opt-jerk-max').value = options.jerk_max;
    if (options.accept_score !== undefined) el('opt-accept-score').value = options.accept_score;
    if (options.scoring_profile !== undefined) {
        el('opt-score-profile').value = ['precision', 'balanced', 'speed']
            .indexOf(options.scoring_profile) >= 0 ? options.scoring_profile : 'balanced';
    }
    if (options.export_format !== undefined) el('opt-export').value = options.export_format;
    if (options.controller_dt !== undefined) el('opt-controller-dt').value = options.controller_dt;
    updateGateFields();
    updateMeshField();
    syncPresetRadio();
}

/* Show the preset that matches the current options, if there is one. The
   preset cards come from the server, which owns every solver flag they set,
   so a CLI preset and a studio preset are the same run. */
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
            && preset.export_format === current.export_format
            && (preset.propagation || 'plant') === current.propagation;
    });
    radios.forEach((radio) => { radio.checked = match ? radio.value === match.value : false; });
}

/* ------------------------------------------------------ mission files */

function missionDocument() {
    const scoring = state.missionScoring || {};
    const clean = {};
    ['target_time_s', 'max_time_s', 'jerk_target_m_s3', 'jerk_max_m_s3'].forEach((key) => {
        if (scoring[key] !== undefined && scoring[key] !== null) clean[key] = scoring[key];
    });
    const waypoints = state.waypoints.map(waypointToJson);
    return Object.keys(clean).length ? { waypoints, scoring: clean } : waypoints;
}

function downloadMission() {
    if (state.waypoints.length < 2) {
        toast('A mission needs at least two waypoints.', true);
        return;
    }
    const name = (el('mission-name').value || 'mission').replace(/[^a-z0-9_-]+/gi, '-');
    const blob = new Blob([JSON.stringify(missionDocument(), null, 1)],
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

function timingName(entry) {
    return entry.timing === 'per-segment' ? 'per-segment' : 'equal-duration';
}

/* The physical identity of a candidate. The planner's three are the frontier
   triad; the rest are the rescue seeds, which are only in the list when the
   triad produced nothing usable (KI-048). */
const CANDIDATE_STYLES = {
    frontier_sprint: { icon: '🚀', label: 'Sprint', detail: 'peak 0.70' },
    frontier_sweeper: { icon: '🌊', label: 'Sweeper', detail: 'peak 0.65' },
    frontier_exploratory: { icon: '🛡️', label: 'Exploratory', detail: '1.25x offset' },
};
/* The styles that are only ever rescue seeds. The current report also carries
   a `rescue` flag, which is what marks a canonical frontier name; this list is
   the fallback for reports written before that flag existed. */
const RESCUE_STYLES = ['baseline', 'rolling', 'bow_left', 'bow_right', 'turn_first'];

function isRescueCandidate(style) {
    return RESCUE_STYLES.indexOf(style) >= 0;
}

function candidateName(style, rescue) {
    const known = CANDIDATE_STYLES[style];
    if (rescue) return `🛟 Rescue: ${known ? known.label : (style || 'baseline')}`;
    if (known) return `${known.icon} ${known.label}`;
    if (isRescueCandidate(style)) return `🛟 Rescue: ${style}`;
    return style || 'baseline';
}

/* The three discretization mechanisms the solver's cost prices, measured on
   the plan that came back (core/optimizer.py's discretization metric): the
   chord-vs-arc sagitta, the parabolic-vs-linear heading mismatch, and the
   20 ms sample-and-hold lag. Each is a distance in mm, against the 5 mm
   tracking gate. */
function segmentMechanisms(from, to) {
    const samples = state.trajectory || [];
    if (from === undefined || to === undefined || to - from < 1
            || from < 0 || to >= samples.length) return null;
    const track = Number(robotFields().track_width) || 0.0965;
    let sagitta = 0;
    let parabolic = 0;
    let hold = 0;
    for (let k = from; k < to; k++) {
        const a = samples[k];
        const b = samples[k + 1];
        const dt = b.t - a.t;
        if (!(dt > 0)) continue;
        const v1 = (a.vl + a.vr) / 2;
        const v2 = (b.vl + b.vr) / 2;
        const w1 = (a.vr - a.vl) / track;
        const w2 = (b.vr - b.vl) / track;
        const vMid = 0.5 * (v1 + v2);
        const wMid = 0.5 * (w1 + w2);
        sagitta = Math.max(sagitta, Math.abs(0.125 * vMid * wMid * dt * dt));
        parabolic = Math.max(parabolic, Math.abs(0.125 * vMid * (w2 - w1) * dt * dt));
        hold = Math.max(hold, Math.abs(0.5 * (v2 - v1) * 0.02));
    }
    return {
        sagitta_mm: sagitta * 1000,
        parabolic_mm: parabolic * 1000,
        hold_mm: hold * 1000,
    };
}

/* A candidate that produced no samples has no duration, and the report
   carries that as null. */
function durationText(entry) {
    const seconds = entry ? entry.duration_s : null;
    if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) {
        return 'an unknown time';
    }
    return `${seconds.toFixed(2)} s`;
}

/* Why the solver picked what it picked: the winner, and what the others cost.
   The rule is said out loud: among the passing candidates the shortest
   duration wins, and the reachability envelope only breaks a tie. */
function describeCandidates(report, winner) {
    const candidates = report.candidates || [];
    const parts = [];

    /* A scored report says the selection rule out loud: the highest score
       wins, and the score is a decision score, never a probability. */
    if (winner && winner.score !== undefined && winner.score !== null) {
        const threshold = report.accept_score;
        parts.push(`the highest score wins: ${Number(winner.score).toFixed(1)} / 100`
            + ` (${winner.profile || report.profile || 'balanced'})`);
        if (threshold !== undefined && threshold !== null) {
            parts.push(`a failed plan is accepted at score ${threshold} or above`);
        }
    }

    if (candidates.length > 4) {
        // More candidates than a chip row can name: summarise them.
        const passed = candidates.filter((entry) => entry.ok).length;
        const failed = candidates.filter((entry) => !entry.ok && !entry.validation_skipped
            && entry.duration_s !== null)
            .sort((a, b) => a.duration_s - b.duration_s);
        const weightOf = (entry) => (entry.accuracy_weight ? ` at weight ${entry.accuracy_weight}` : '');
        if (winner) {
            parts.push(`${candidates.length} candidates: ${timingName(winner)}`
                + `${weightOf(winner)} won in ${durationText(winner)}, ${passed} passed`);
        }
        if (failed.length) {
            const best = failed[0];
            parts.push(`the fastest that failed was ${timingName(best)}`
                + `${weightOf(best)} in ${durationText(best)}: `
                + `${(best.reasons || []).join('; ')}`);
        }
        parts.push('the winner is the passing candidate with the shortest time; '
            + 'the envelope only breaks a tie');
        return parts.join('. ');
    }

    if (winner) {
        const duration = winner.duration_s === null || winner.duration_s === undefined
            ? '' : ` (${winner.duration_s.toFixed(2)} s)`;
        const lead = winner.ok
            ? `${timingName(winner)} ${winner.style || 'baseline'} won${duration}`
            : `no candidate passed; ${timingName(winner)} ${winner.style || 'baseline'} came closest${duration}`;
        parts.push(lead);
        if (winner.ok) {
            parts.push('the winner is the passing candidate with the shortest time, '
                + 'and the envelope only breaks a tie');
        }
    }
    candidates.forEach((entry) => {
        if (winner && entry === winner) return;
        const other = entry.duration_s;
        const mine = winner ? winner.duration_s : null;
        let difference = '';
        let sameTime = false;
        if (other !== null && other !== undefined && mine !== null && mine !== undefined) {
            const delta = Math.abs(mine - other);
            sameTime = delta < 0.005;
            difference = sameTime
                ? ' took the same time'
                : (other < mine
                    ? ` was ${delta.toFixed(2)} s faster` : ` was ${delta.toFixed(2)} s slower`);
        }
        const why = entry.reasons && entry.reasons.length ? `: ${entry.reasons.join('; ')}` : '';
        /* A candidate that passed but did not win needs the reason, or the
           reader is left with a faster one that lost. */
        let tail = entry.validation_skipped ? '' : why;
        if (entry.ok && winner && winner.ok && other !== null && other !== undefined
                && mine !== null && mine !== undefined && other <= mine) {
            const hasEnvelope = winner.envelope_cm !== null && winner.envelope_cm !== undefined
                && entry.envelope_cm !== null && entry.envelope_cm !== undefined;
            /* Two candidates that are the same to the eye still differ: the
               envelope decides, and the numbers are given to three places so
               the difference can be seen. */
            const apart = hasEnvelope
                && Math.abs(entry.envelope_cm - winner.envelope_cm) > 0.0005;
            tail = apart
                ? `, but its envelope was ${entry.envelope_cm.toFixed(3)} cm against `
                    + `${winner.envelope_cm.toFixed(3)} cm, so it did not win`
                : (hasEnvelope
                    ? ', and the tie went to the one that came first'
                    : ', but it did not win');
        }
        parts.push(`${timingName(entry)} ${entry.style || 'baseline'}${difference} `
            + `${entry.validation_skipped ? 'and was not checked'
                : (entry.ok ? 'and passed' : 'but failed')}${tail}`);
    });
    return parts.join('. ');
}

/* How far a best-effort path missed, in the words the operator reads. The
   number is the biggest per-segment excess over the local limit; the reasons
   name the gate, so "failed" is never a bare word. */
function bestEffortDetail(candidate) {
    const excess = candidate && candidate.max_excess_cm;
    const reasons = ((candidate && candidate.reasons) || []).filter(Boolean);
    let lead = 'it did not pass the validator';
    if (excess !== null && excess !== undefined && excess > 0) {
        lead = `it is up to ${excess.toFixed(2)} cm outside the local limit`;
    }
    return lead[0].toUpperCase() + lead.slice(1)
        + (reasons.length ? ` (${reasons.join('; ')})` : '') + '.';
}

/* The per-sample accuracy limit the field imposed (--field-clearance),
   shown next to the accuracy limit it replaces. The tooltip carries the
   field's placement uncertainty: a tight limit read off FIRST's wireframes
   is nominal geometry, never a survey. */
function updateFieldLimitNote(report) {
    const note = el('field-limit-note');
    const field = report && report.envelope_limit_field_m;
    if (!field) {
        note.hidden = true;
        note.textContent = '';
        note.title = '';
        return;
    }
    const cm = (value) => (Number(value) * 100).toFixed(1);
    const hasRange = field.min_m !== null && field.min_m !== undefined
        && field.max_m !== null && field.max_m !== undefined;
    note.hidden = false;
    note.textContent = hasRange
        ? `Field ${field.season}: ${cm(field.min_m)}-${cm(field.max_m)} cm per sample`
        : `Field ${field.season}: per-sample limits`;
    const unsure = Number(report.placement_uncertainty_m);
    note.title = 'The accuracy limit becomes the field clearance at each sample, '
        + 'clamped to 0.5-5 cm; a sample the field cannot measure falls back to '
        + 'the limit on the left. The field footprints agree only to about '
        + `${Number.isFinite(unsure) ? (unsure * 100).toFixed(0) : '100'} mm`
        + ' (placement uncertainty), so this limit is nominal geometry, '
        + 'not a survey.';
}

/* The verdict of the last run. This is the one place that owns it: a report
   draws it, and no report takes it away again. A solve without validation
   has no verdict, so the screen must not keep the one before it. */
function renderValidationReport(report) {
    state.validationReport = report || null;
    updateFieldLimitNote(report);
    const box = el('validation-report');
    if (!report || !report.candidates || !report.candidates.length) {
        /* No verdict: everything the last one left on the screen goes, so
           a new run cannot be read through the old one. */
        box.hidden = true;
        el('validation-candidates').innerHTML = '';
        el('validation-table').innerHTML = '';
        el('validation-explainer').textContent = '';
        el('validation-note').textContent = '';
        el('fix-row').innerHTML = '';
        el('fix-row').hidden = true;
        el('pick-row').innerHTML = '';
        el('pick-row').hidden = true;
        state.reportCandidate = null;
        state.failingSegments = new Set();
        state.worstSegment = -1;
        /* The red segments on the field are the same verdict, so they go
           with it, and the key drops the model it named. */
        updateLegendLabels();
        draw();
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
        /* The seed is in the chip, not only in the tooltip: the fallback
           seeds would otherwise be indistinguishable. The planner's three are
           the frontier triad; a rescue seed is marked, because it is only in
           the list when the triad failed (KI-048). */
        const style = entry.style || 'baseline';
        const rescue = Boolean(entry.rescue) || isRescueCandidate(style);
        const verdict = entry.ok ? 'ok'
            : (entry.accepted ? 'accepted'
                : (entry.validation_skipped ? 'not checked' : 'failed'));
        chip.className = `chip ${entry.ok ? 'ok' : (entry.accepted ? 'warn' : (entry.validation_skipped ? '' : 'bad'))}`
            + `${i === index ? ' active' : ''}`
            + (rescue ? ' is-rescue' : '');
        const timing = entry.timing === 'per-segment' ? 'per-seg' : 'equal-dur';
        const duration = entry.duration_s === null || entry.duration_s === undefined
            ? '' : ` ${entry.duration_s.toFixed(2)}s`;
        const weight = (entry.accuracy_weight !== null && entry.accuracy_weight !== undefined
            && entry.accuracy_weight > 0) ? ` @${entry.accuracy_weight}` : '';
        const score = entry.score === null || entry.score === undefined
            ? '' : ` score ${Number(entry.score).toFixed(1)}`;
        const eligibility = entry.eligibility ? ` ${entry.eligibility}` : '';
        chip.textContent = `${i + 1}. ${candidateName(style, rescue)} ${timing}${weight}${duration}`
            + `${score}${eligibility} ${verdict}`;
        // One title: the candidate's identity and why it failed. The second
        // assignment used to overwrite the first, so the tooltip never
        // showed the style or the weight.
        const reasons = (entry.reasons || []).join('; ');
        /* A cut-off propagation reports the width the tube had when it
           crossed the validator's 50 mm guard, not the width it would have
           reached, so the number is a property of the guard. Say so, or the
           operator reads a truncation as a measurement (KI-049). */
        const truncated = entry.envelope_truncated
            ? ' (cut off at the guard, so it is a lower bound)' : '';
        /* The envelope is the width of a reachability tube, and the tube is
           a property of the propagation, so the model belongs next to the
           number (KI-049). */
        const model = entry.propagation === 'tick'
            ? 'closed-loop tick map' : 'open-loop plant';
        chip.title = `${entry.style || 'baseline'}, ${timing}, `
            + `weight ${entry.accuracy_weight === null ? '-' : entry.accuracy_weight}`
            + (entry.envelope_cm === null || entry.envelope_cm === undefined
                ? '' : `, envelope ${entry.envelope_cm.toFixed(2)} cm${truncated}`
                    + ` (${model})`)
            + (reasons ? `: ${reasons}` : '');
        chip.addEventListener('click', () => {
            state.reportCandidate = i;
            renderValidationReport(report);
        });
        chips.appendChild(chip);
    });

    // The trade-off between the candidates, in words.
    const winner = report.candidates[(report.selected || 1) - 1] || report.candidates[0];
    const explainer = el('validation-explainer');
    explainer.textContent = describeCandidates(report, winner);
    renderPick(report, index);
    renderFixes(report);

    const table = el('validation-table');
    table.innerHTML = '';
    const header = document.createElement('tr');
    /* The last three columns are the discretization mechanisms the solver's
       cost prices, measured on the plan that came back: the sagitta of the
       chord, the heading mismatch of the parabolic step and the 20 ms
       sample-and-hold lag. They are distances in mm, against the 5 mm
       tracking gate, so a table row says why a segment tracked the way it
       did. */
    ['Segment', 'Envelope', 'Limit', 'Drift', 'Sagitta', 'Heading', 'Hold'].forEach((label) => {
        const th = document.createElement('th');
        th.textContent = label;
        header.appendChild(th);
    });
    table.appendChild(header);

    const bounds = segmentBounds();
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
        const mechanism = segmentMechanisms(bounds[segment.from], bounds[segment.to]);
        const cell = (value) => (mechanism ? `${value.toFixed(2)} mm` : '-');
        row.innerHTML = `<td>W${segment.from} &rarr; W${segment.to}</td>`
            + `<td>${envelope.toFixed(2)} cm</td>`
            + `<td>${limit.toFixed(2)} cm</td>`
            + `<td>${drift.toFixed(2)} cm</td>`
            + `<td>${cell(mechanism ? mechanism.sagitta_mm : 0)}</td>`
            + `<td>${cell(mechanism ? mechanism.parabolic_mm : 0)}</td>`
            + `<td>${cell(mechanism ? mechanism.hold_mm : 0)}</td>`;
        if (!segment.ok) row.className = 'bad';
        else if (worst === segment.from && worstDrift > 0.005) row.className = 'warn';
        table.appendChild(row);
    });
    state.failingSegments = failing;
    state.worstSegment = worst;

    const notes = [];
    /* Which model produced the envelope. The tube is a property of the
       propagation, and the two propagations answer different questions: the
       plant propagation bounds an open-loop plant driven by a pre-recorded
       command, and the tick map bounds the closed loop the robot actually
       runs. An envelope without its model cannot be judged (KI-049). */
    const propagation = candidate.propagation || report.propagation || 'plant';
    notes.push(propagation === 'tick'
        ? 'the envelope is the closed-loop tick map\'s tube: the feedback contracts it'
        : 'the envelope is the open-loop plant\'s tube: the controller is outside it');
    if (candidate.tracking_cm !== null && candidate.tracking_cm !== undefined) {
        notes.push(`the robot tracked the path to within ${candidate.tracking_cm.toFixed(2)} cm `
            + `(the limit is 0.50 cm)`);
    }
    if (candidate.reasons && candidate.reasons.length) {
        notes.push(candidate.reasons.join('; '));
    }
    if (worst >= 0 && worstDrift > 0.005) {
        notes.push(`the path drifts most between W${worst} and W${worst + 1}`);
    }
    el('validation-note').textContent = notes.join('. ');
    /* The key names the model the envelope came from, so it follows the
       verdict rather than the page load. */
    updateLegendLabels();
    draw();
}

/* ------------------------------------------------------------- the report

   The one-page report. It is built from the same numbers the Result tab
   shows, so it cannot disagree with the studio, and it is a single
   self-contained document: printable, and downloadable as it stands. */

function collectReportData() {
    const summary = state.jobSummary || {};
    const report = state.validationReport;
    const selected = report && report.candidates
        ? ((report.selected || 1) - 1) : -1;
    const candidate = report && report.candidates
        ? (report.candidates[state.reportCandidate !== null
            && state.reportCandidate < report.candidates.length
            ? state.reportCandidate : selected] || null)
        : null;
    const events = state.waypoints.filter((point) => point.event).length;
    return {
        mission: {
            name: el('mission-name').value,
            waypoints: state.waypoints.map(waypointToJson),
            scoring: state.missionScoring || {},
        },
        robot: {
            name: state.configName,
            fields: robotFields(),
            model: state.model,
            useModel: Boolean(state.useModel),
        },
        field: {
            season: state.season,
            widthM: state.mat.widthM,
            heightM: state.mat.heightM,
        },
        options: collectOptions(),
        result: {
            when: state.jobFinishedAt || new Date().toISOString(),
            summary,
            samples: state.trajectory || [],
            candidates: report ? report.candidates : null,
            selected: candidate ? report.candidates.indexOf(candidate) : -1,
            segments: candidate ? (candidate.segments || []) : null,
            tracking_cm: candidate ? candidate.tracking_cm : null,
            validated: el('opt-validate').checked,
            failed: Boolean(state.jobFailed),
            outcome: state.runOutcome,
            score: state.jobScore,
            score_version: state.jobScore === null ? null : 1,
            profile: el('opt-score-profile').value,
            field_clearance_util: candidate ? candidate.field_clearance_util : null,
            field_clearance_min_m: candidate ? candidate.field_clearance_min_m : null,
            penetration_m: candidate ? candidate.penetration_m : null,
            field_clearance_skipped_samples: candidate
                ? candidate.field_clearance_skipped_samples : null,
            hub_approved: state.hubApproved,
            accepted: state.runOutcome === 'accepted',
            reason: state.jobFailedReason || '',
            wall_s: state.jobWallS,
            path_length_m: summary.path_length_m !== undefined
                ? summary.path_length_m : pathLength(),
            min_time_s: minTime(),
            events,
        },
    };
}

/* The fastest the robot could do the path at its top speed: the floor the
   plan is measured against. */
function minTime() {
    const speed = topSpeed();
    if (!speed) return null;
    const length = pathLength();
    return length > 0 ? length / speed : null;
}

function openMissionReport() {
    if (state.waypoints.length < 2) {
        toast('A report needs a mission with at least two waypoints.', true);
        return;
    }
    const html = buildMissionReport(collectReportData());
    // The tab is opened here, in the click, because a browser only allows a
    // pop-up while the click is still the thing being handled.
    const view = window.open('', '_blank');
    if (!view) {
        toast('The report window was blocked; allow pop-ups for this page.', true);
        return;
    }
    if (!state.jobId) {
        showReportIn(view, html);
        return;
    }
    // The report is written next to the trajectory, so it is on disk with
    // the rest of the solve.
    api(`/api/jobs/${state.jobId}/report`, {
        method: 'POST',
        body: JSON.stringify({ html }),
    }).then((result) => {
        view.location.href = result.url;
        addReportDownload(result);
    }).catch((err) => {
        // The file could not be written, but the page can still be shown.
        showReportIn(view, html);
        toast(err.message, true);
    });
}

function showReportIn(view, html) {
    view.document.open();
    view.document.write(html);
    view.document.close();
}

/* The report joins the other downloads of this solve. */
function addReportDownload(result) {
    const box = el('downloads');
    if (!box || !result) return;
    if (box.querySelector('[data-kind="report"]')) return;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'btn small';
    button.dataset.kind = 'report';
    button.textContent = 'Report (one page)';
    button.addEventListener('click', () => {
        window.open(result.url, '_blank');
    });
    box.appendChild(button);
}

/* ------------------------------------------------------ the project file

   One file with the mission, the robot, the solve options and the measured
   model: a setup that can be handed to somebody else and opened as it
   stands. */

const PROJECT_FORMAT = 'overboard-project';
const PROJECT_VERSION = 1;

function collectProject() {
    return {
        format: PROJECT_FORMAT,
        version: PROJECT_VERSION,
        saved: new Date().toISOString(),
        mission: {
            name: el('mission-name').value,
            waypoints: state.waypoints.map(waypointToJson),
            scoring: state.missionScoring || {},
        },
        robot: {
            name: state.configName,
            config: state.config,
        },
        options: collectOptions(),
        model: state.useModel ? state.model : null,
        field: {
            season: state.season,
            width_m: state.mat.widthM,
            height_m: state.mat.heightM,
        },
    };
}

function downloadProject() {
    if (state.waypoints.length < 2) {
        toast('A project needs a mission with at least two waypoints.', true);
        return;
    }
    const name = (el('mission-name').value || 'project').replace(/[^a-z0-9_-]+/gi, '-');
    const blob = new Blob([JSON.stringify(collectProject(), null, 1)],
        { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `${name}.overboard.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    toast(`Downloaded ${name}.overboard.json`);
}

/* Open a project file: the mission, the robot, the options and the model
   all come back. A file that is not a project is refused with a reason. */
function openProjectFile(file) {
    const reader = new FileReader();
    reader.onload = () => {
        let data;
        try {
            data = JSON.parse(String(reader.result));
        } catch (err) {
            toast('That file is not JSON.', true);
            return;
        }
        if (!data || data.format !== PROJECT_FORMAT) {
            toast('That file is not an OVERBOARD project.', true);
            return;
        }
        try {
            if (data.mission && Array.isArray(data.mission.waypoints)) {
                setWaypoints(data.mission.waypoints);
                el('mission-name').value = data.mission.name || 'mission';
            }
            if (data.robot && data.robot.config) {
                state.config = data.robot.config;
                state.configName = data.robot.name || state.configName;
                state.configEdited = false;
                renderRobotFacts();
                renderRobotEditor();
                renderRobotChecks();
            }
            if (data.options) {
                applyOptionsToForm(data.options);
                if (data.options.sysid_fit) {
                    const path = data.options.sysid_fit;
                    const found = (state.models || []).find((model) => model.path === path);
                    if (found) useModel(found);
                }
            }
            if (data.model) {
                state.model = data.model;
                state.useModel = true;
                el('opt-use-model').checked = true;
                renderModel();
                updateGateFields();
            }
            if (data.field) {
                if (data.field.season) {
                    state.season = data.field.season;
                    const select = el('field-season');
                    if (select) select.value = data.field.season;
                }
                if (Number.isFinite(data.field.width_m) && data.field.width_m > 0) {
                    state.mat.widthM = data.field.width_m;
                    el('mat-width').value = String(Math.round(data.field.width_m * 1000));
                }
                if (Number.isFinite(data.field.height_m) && data.field.height_m > 0) {
                    state.mat.heightM = data.field.height_m;
                    el('mat-height').value = String(Math.round(data.field.height_m * 1000));
                }
            }
            resetHistory();
            frameAll();
            draw();
            scheduleSessionSave();
            toast('Project opened.');
        } catch (err) {
            toast(err.message, true);
        }
    };
    reader.readAsText(file);
}

/* ------------------------------------------------------- run on the robot */

/* The robot's odometry starts where the robot physically is, which is the
 * plan's first waypoint. The measured pose is therefore rotated by the plan's
 * first heading and moved by its first position before it can be drawn next
 * to the plan. */
function measuredInPlanFrame(pose) {
    if (!state.trajectory || !state.trajectory.length) return null;
    const first = state.trajectory[0];
    const heading = Number(first.heading) || 0;
    const cos = Math.cos(heading);
    const sin = Math.sin(heading);
    return {
        x: Number(first.x) + Number(pose.x_m) * cos - Number(pose.y_m) * sin,
        y: Number(first.y) + Number(pose.x_m) * sin + Number(pose.y_m) * cos,
        heading_deg: (Number(pose.heading_deg) || 0) + heading * 180 / Math.PI,
    };
}

function pushSummary() {
    /* The last planned pose and the measured one, in the plan's frame. */
    if (!state.trajectory || !state.trajectory.length) return null;
    const last = state.trajectory[state.trajectory.length - 1];
    const measured = state.measuredPoses.length
        ? measuredInPlanFrame(state.measuredPoses[state.measuredPoses.length - 1])
        : null;
    if (!measured) return null;
    const error = Math.hypot(measured.x - last.x, measured.y - last.y);
    let headingError = Math.abs(measured.heading_deg - (last.heading * 180 / Math.PI)) % 360;
    if (headingError > 180) headingError = 360 - headingError;
    const env = last.reachability_envelope;
    let inside = null;
    if (env) {
        inside = (measured.x >= env.x_min && measured.x <= env.x_max
            && measured.y >= env.y_min && measured.y <= env.y_max);
    }
    return {
        error_mm: error * 1000,
        heading_error_deg: headingError,
        measured,
        inside,
    };
}

function renderPushResult(job) {
    const result = job.result || {};
    const box = el('push-result');
    const facts = el('push-facts');
    facts.innerHTML = '';
    const rows = [];

    // The poses come first: the summary is built from them, so it has to
    // read them after they are stored and not before.
    state.measuredPoses = result.poses || [];

    const summary = pushSummary();
    if (summary) {
        rows.push(['Robot ended', `${summary.measured.x.toFixed(3)}, ${summary.measured.y.toFixed(3)} m`]);
        rows.push(['Off the plan', `${summary.error_mm.toFixed(1)} mm`]);
        rows.push(['Heading error', `${summary.heading_error_deg.toFixed(2)} deg`]);
        if (summary.inside !== null) {
            rows.push(['Inside envelope', summary.inside ? 'yes' : 'NO']);
        }
    }
    if (result.events && result.events.length) {
        rows.push(['Events', result.events.join(', ')]);
    }
    if (result.session_path) {
        rows.push(['Session', result.session_path]);
    }
    rows.forEach(([key, value]) => {
        const dt = document.createElement('dt');
        dt.textContent = key;
        const dd = document.createElement('dd');
        dd.textContent = value;
        facts.appendChild(dt);
        facts.appendChild(dd);
    });
    box.hidden = rows.length === 0;
    el('push-events').textContent = result.events && result.events.length
        ? `The hub reported: ${result.events.join(', ')}.`
        : '';

    renderComparison(result.report);
    draw();
}

/* The plan against what the robot did, in the same table the command line
 * prints. */
function renderComparison(report) {
    const box = el('compare-block');
    if (!report || report.error || !report.rows || !report.rows.length) {
        box.hidden = true;
        el('fix-row').hidden = true;
        return;
    }
    box.hidden = false;
    const table = el('compare-table');
    table.innerHTML = '';
    const header = document.createElement('tr');
    ['Stop', 't (s)', 'Measured x/y (mm)', 'Planned x/y (mm)', 'Error (mm)',
        'Heading (deg)', 'Envelope'].forEach((label) => {
        const th = document.createElement('th');
        th.textContent = label;
        header.appendChild(th);
    });
    table.appendChild(header);

    report.rows.forEach((row) => {
        const tr = document.createElement('tr');
        const cells = [
            String(row.stop),
            row.t.toFixed(3),
            `${(row.measured.x * 1000).toFixed(1)} / ${(row.measured.y * 1000).toFixed(1)}`,
            `${(row.predicted.x * 1000).toFixed(1)} / ${(row.predicted.y * 1000).toFixed(1)}`,
            (row.position_error_m * 1000).toFixed(2),
            (row.heading_error_rad * 180 / Math.PI).toFixed(2),
        ];
        cells.forEach((value) => {
            const td = document.createElement('td');
            td.textContent = value;
            tr.appendChild(td);
        });
        const envelope = document.createElement('td');
        if (row.inside_envelope === null) envelope.textContent = '-';
        else if (row.inside_envelope) envelope.textContent = 'inside';
        else { envelope.textContent = 'OUTSIDE'; tr.className = 'bad'; }
        tr.appendChild(envelope);
        table.appendChild(tr);
    });

    const notes = [];
    notes.push(`Max position error ${(report.max_position_error_m * 1000).toFixed(2)} mm`);
    notes.push(`RMS ${(report.rms_position_error_m * 1000).toFixed(2)} mm`);
    notes.push(`max heading error ${(report.max_heading_error_rad * 180 / Math.PI).toFixed(2)} deg`);
    (report.geometry_warnings || []).forEach((warning) => notes.push(warning));
    el('compare-note').textContent = notes.join(' | ');
}

async function scanHubs() {
    /* The scan asks every device around whether it speaks the Pybricks
       protocol, so it takes about a minute: the status says so, and the
       button is disabled rather than looking dead. */
    const button = el('btn-scan');
    button.disabled = true;
    el('push-status').textContent = 'Scanning for hubs over Bluetooth. This takes about a minute.';
    try {
        const found = await api('/api/hubs', { method: 'POST' });
        const hubs = found.hubs || [];
        const choices = el('hub-choices');
        choices.textContent = '';
        hubs.forEach((hub) => {
            const option = document.createElement('option');
            option.value = hub.name;
            option.label = hub.label;
            choices.appendChild(option);
        });
        if (!hubs.length) {
            el('push-status').textContent = `No Pybricks hub answered among `
                + `${found.seen} device(s). Turn the hub on, close Pybricks Code, `
                + 'and scan again.';
            toast('No Pybricks hub answered.', true);
            return;
        }
        if (hubs.length === 1) {
            el('hub-name').value = hubs[0].name;
            state.hubName = hubs[0].name;
            scheduleSessionSave();
        }
        el('push-status').textContent = hubs.length === 1
            ? `Found ${hubs[0].label}.`
            : `Found ${hubs.length} hubs. The Hub name field lists them.`;
        toast(`Found ${hubs.length} hub(s).`);
    } catch (err) {
        el('push-status').textContent = err.message;
        toast(err.message, true);
    } finally {
        button.disabled = false;
    }
}

async function pushToHub() {
    if (!state.solveJobId || !state.solveHasTrajectory) {
        toast('Solve a mission first: the push runs an approved trajectory.', true);
        return;
    }
    /* What the server allows is what the button allows: a validated pass,
       or a failed plan accepted by its score and marked hub-ready. The
       server re-checks the file's approval either way. */
    const approved = state.solveValidated
        || (state.runOutcome === 'accepted' && state.hubApproved === true);
    if (!approved) {
        toast('This trajectory has not been validated. Validate it before pushing it '
            + 'to the robot.', true);
        return;
    }
    if (state.dirty) {
        toast('The mission changed since this solve - solve again before pushing it.', true);
        return;
    }
    setStatus(`Pushing to the hub over ${state.hubConn.toUpperCase()} - the robot is driving`);
    el('push-status').textContent = 'Pushing. The robot is driving.';
    el('btn-push').disabled = true;
    el('log').textContent = '';
    state.pushPollRetries = 0;
    try {
        const job = await api('/api/push', {
            method: 'POST',
            body: JSON.stringify({
                job_id: state.solveJobId,
                connection: state.hubConn,
                hub_name: state.hubName,
            }),
        });
        state.pushJobId = job.id;
        pollPush();
    } catch (err) {
        el('push-status').textContent = err.message;
        setStatus('Push failed', 'error');
        toast(err.message, true);
        updatePushButton();
    }
}

function pollPush() {
    clearTimeout(state.pushPollTimer);
    state.pushPollTimer = setTimeout(async () => {
        try {
            const job = await api(`/api/jobs/${state.pushJobId}`);
            el('log').textContent = (job.log || []).join('\n');
            if (job.status === 'running') {
                pollPush();
                return;
            }
            if (job.status === 'ok') {
                el('push-status').textContent = 'The hub ran the mission.';
                setStatus('Pushed', 'ok');
                renderPushResult(job);
                renderDownloads(job.artifacts, job.id);
                toast('The hub ran the mission.');
            } else {
                el('push-status').textContent = job.error || 'The push failed.';
                setStatus('Push failed', 'error');
                toast(job.error || 'The push failed.', true);
            }
            updatePushButton();
        } catch (err) {
            if (state.pushJobId && state.pushPollRetries < 3) {
                state.pushPollRetries += 1;
                pollPush();
                return;
            }
            setStatus(err.message, 'error');
            toast(err.message, true);
            updatePushButton();
        }
    }, 400);
}

function updatePushButton() {
    /* Only a checked trajectory goes to the robot. A solve that was never
       validated has no verdict, so it is not an approved one, and the
       button says so instead of calling it approved. */
    const solved = Boolean(state.solveJobId && state.solveHasTrajectory);
    const checked = state.solveValidated
        || (state.runOutcome === 'accepted' && state.hubApproved === true);
    /* The trajectory on the field has to be the answer to the mission on
       the screen. When the mission has moved on since the solve, the
       trajectory answers an older mission and must not go to the robot. */
    const ready = solved && !state.dirty;
    el('btn-push').disabled = !ready || !checked;
    const status = el('push-status');
    if (solved && state.dirty) {
        status.textContent = 'The mission changed since this solve - solve again '
            + 'before pushing it.';
    } else if (!solved) {
        status.textContent = 'Solve a mission first.';
    } else if (state.runOutcome === 'accepted' && state.hubApproved === true) {
        status.textContent = 'This accepted plan is marked hub-ready by the score. '
            + 'It failed validation - push it only if that is the plan you mean to run.';
    } else if (!checked && state.validationReport) {
        status.textContent = 'This trajectory failed validation, so it is not '
            + 'approved for the robot. It is on the field for inspection only.';
    } else if (!checked) {
        status.textContent = 'This trajectory has not been validated against your robot. '
            + 'Turn validation on in the Solve tab and solve again before pushing it.';
    } else if (!status.textContent || status.textContent.startsWith('Solve a mission')
            || status.textContent.startsWith('This trajectory')
            || status.textContent.startsWith('This accepted plan')) {
        status.textContent = 'The validated trajectory is ready to run on the robot.';
    }
}

/* --------------------------------------------- the robot's own files */

function renderRobotChecks() {
    const info = state.robotInfo || { events: [], geometry: {} };
    const robot = robotFields();
    const geometry = info.geometry || {};

    // The packed mission carries the planned geometry to the hub, while the
    // drivebase comes from the robot's own file. A mismatch means the
    // controller is tuned for a wheel the robot does not have.
    const warning = el('robot-geometry-warning');
    const fix = el('robot-geometry-fix');
    const plannedDiameter = (Number(robot.wheel_radius) || 0) * 2000;
    const plannedTrack = (Number(robot.track_width) || 0) * 1000;
    const differences = [];
    if (geometry.WHEEL_DIAMETER_MM && plannedDiameter
        && Math.abs(geometry.WHEEL_DIAMETER_MM - plannedDiameter) > 0.5) {
        differences.push(`the plan assumes a ${plannedDiameter.toFixed(1)} mm wheel, `
            + `the robot has ${geometry.WHEEL_DIAMETER_MM.toFixed(1)} mm`);
    }
    if (geometry.AXLE_TRACK_MM && plannedTrack
        && Math.abs(geometry.AXLE_TRACK_MM - plannedTrack) > 0.5) {
        differences.push(`the plan assumes a ${plannedTrack.toFixed(1)} mm track, `
            + `the robot has ${geometry.AXLE_TRACK_MM.toFixed(1)} mm`);
    }
    warning.hidden = differences.length === 0;
    fix.hidden = differences.length === 0;
    if (differences.length) {
        warning.textContent = `The robot's own file disagrees with the plan: `
            + `${differences.join('; ')}.`;
    }

    // The event names the robot has actions for.
    const eventWarning = el('robot-event-warning');
    const used = state.waypoints.map((p) => p.event).filter(Boolean);
    const unknown = used.filter((name) => !(info.events || []).includes(name));
    eventWarning.hidden = unknown.length === 0;
    if (unknown.length) {
        eventWarning.textContent = `The robot has no action for `
            + `${unknown.join(', ')}. The hub will report the event and carry on.`;
    }
}

function useRobotGeometry() {
    const geometry = (state.robotInfo || {}).geometry || {};
    const fields = robotFields();
    if (geometry.WHEEL_DIAMETER_MM) {
        fields.wheel_radius = geometry.WHEEL_DIAMETER_MM / 2000;
    }
    if (geometry.AXLE_TRACK_MM) {
        fields.track_width = geometry.AXLE_TRACK_MM / 1000;
    }
    setRobotFields(fields);
    markMissionChanged();
    renderRobotFacts();
    renderRobotEditor();
    renderRobotChecks();
    updatePathSummary();
    scheduleSessionSave();
}

/* ------------------------------------------------- the measured model */

/* The fit that the plan is checked against. It belongs to the robot, not to
 * the mission: one calibration makes every later mission better. */
function renderModel() {
    const facts = el('model-facts');
    facts.innerHTML = '';
    const model = state.model;
    const status = el('model-status');
    if (!model) {
        status.textContent = 'No measured model: the plan uses the nominal one.';
        el('model-hint').hidden = true;
        return;
    }
    status.textContent = `Loaded from ${model.path}.`;

    const rows = [];
    const gain = (entry) => {
        if (!entry || !entry.gain) return null;
        const ci = entry.gain_ci95
            ? ` [${entry.gain_ci95[0].toFixed(3)}, ${entry.gain_ci95[1].toFixed(3)}]`
            : '';
        return `${entry.gain.toFixed(3)}${ci}`;
    };
    if (model.straight && model.straight.calibrated) {
        rows.push(['Straight gain', gain(model.straight)]);
        rows.push(['Straight tau / delay',
            `${model.straight.tau_s?.toFixed(3)} s / ${model.straight.delay_s?.toFixed(3)} s`]);
    }
    if (model.turn && model.turn.calibrated) {
        rows.push(['Turn gain', gain(model.turn)]);
        rows.push(['Turn tau / delay',
            `${model.turn.tau_s?.toFixed(3)} s / ${model.turn.delay_s?.toFixed(3)} s`]);
    }
    if (model.gyro_z_bias_deg_s !== null && model.gyro_z_bias_deg_s !== undefined) {
        rows.push(['Gyro z bias', `${model.gyro_z_bias_deg_s.toFixed(4)} deg/s`]);
    }
    if (model.odometry && model.odometry.calibrated) {
        const ci = model.odometry.distance_scale_ci95
            ? ` [${model.odometry.distance_scale_ci95[0].toFixed(4)}, ${model.odometry.distance_scale_ci95[1].toFixed(4)}]`
            : '';
        rows.push(['Distance scale', `${model.odometry.distance_scale?.toFixed(4)}${ci}`]);
        rows.push(['Heading scale', model.odometry.heading_scale?.toFixed(4)]);
    }
    if (model.slip && model.slip.calibrated) {
        rows.push(['Brake slip', `${model.slip.brake_ratio_median?.toFixed(4)} `
            + `[${model.slip.brake_ratio_min?.toFixed(4)}, ${model.slip.brake_ratio_max?.toFixed(4)}]`]);
    }
    rows.forEach(([key, value]) => {
        const dt = document.createElement('dt');
        dt.textContent = key;
        const dd = document.createElement('dd');
        dd.textContent = value === null || value === undefined ? '-' : value;
        facts.appendChild(dt);
        facts.appendChild(dd);
    });

    // The gain band is the band the validator plans against, so a robot that
    // varies by a percent is a percent of position error over the path. Say so
    // before the solve fails on it.
    const hint = el('model-hint');
    const ci = model.straight && model.straight.gain_ci95;
    if (ci && ci.length === 2) {
        const spread = Math.max(Math.abs(1 - ci[0]), Math.abs(1 - ci[1]));
        hint.textContent = `The gain band is ${(spread * 100).toFixed(1)}%. The controller `
            + `corrects most of that speed error, so the tracking barely moves, but the `
            + `reachable set widens with it and the envelope limit is 20 mm. See `
            + `docs/research/RQ-A_tracking_error.md.`;
        hint.hidden = false;
    } else {
        hint.hidden = true;
    }
}

function useModel(model) {
    state.model = model;
    state.useModel = Boolean(model);
    el('opt-use-model').checked = state.useModel;
    if (state.useModel) {
        // The fitted plant is only used by the validator, so validation is on.
        el('opt-validate').checked = true;
        updateGateFields();
    }
    renderModel();
    scheduleSessionSave();
}

async function loadModels() {
    const data = await api('/api/robot/models');
    const list = el('model-list');
    list.innerHTML = '';
    (data.models || []).forEach((model) => {
        const option = document.createElement('option');
        option.value = model.path;
        option.textContent = model.name;
        list.appendChild(option);
    });
    el('model-list-row').hidden = !(data.models || []).length;
}

async function loadModelFromPath() {
    const path = el('model-path').value.trim();
    if (!path) {
        toast('Give the path of a sysid_fit.json.', true);
        return;
    }
    try {
        const model = await api('/api/robot/model/load', {
            method: 'POST',
            body: JSON.stringify({ path }),
        });
        useModel(model);
        await loadModels();
        toast(`Loaded the measured model from ${path}.`);
    } catch (err) {
        toast(err.message, true);
    }
}

async function loadModelFromFile(file) {
    try {
        const fit = JSON.parse(await file.text());
        const model = await api('/api/robot/model/store', {
            method: 'POST',
            body: JSON.stringify({ name: file.name.replace(/\.json$/i, ''), fit }),
        });
        useModel(model);
        await loadModels();
        toast('Stored the measured model with the robot.');
    } catch (err) {
        toast(err.message, true);
    }
}

/* ------------------------------------------------- the calibration wizard */

/* The three steps of the calibration: collect, measure, fit. The steps use the
 * same tools the command line uses, so a session collected here can also be
 * fitted with sysid_fit.py. */
function calibrationName() {
    return (el('cal-name').value || 'robot').trim() || 'robot';
}

function renderSheet(payload) {
    const table = el('sheet-table');
    table.innerHTML = '';
    const rows = payload.rows || [];
    if (!rows.length) {
        el('sheet-status').textContent = 'No measurements yet. Collect a session first.';
        return;
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
        const key = document.createElement('td');
        key.textContent = row.test;
        const repeat = document.createElement('td');
        repeat.textContent = row.repeat;
        const stop = document.createElement('td');
        stop.textContent = row.stop;
        tr.appendChild(key);
        tr.appendChild(repeat);
        tr.appendChild(stop);
        ['x_mm', 'y_mm', 'turn_right_deg'].forEach((field) => {
            const td = document.createElement('td');
            const input = document.createElement('input');
            input.type = 'number';
            input.step = 'any';
            input.value = row[field] === undefined || row[field] === null ? '' : row[field];
            input.dataset.test = row.test;
            input.dataset.repeat = row.repeat;
            input.dataset.stop = row.stop;
            input.dataset.field = field;
            input.addEventListener('input', () => { row[field] = input.value; });
            td.appendChild(input);
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });

    const missing = payload.missing || 0;
    // A measurement is only usable when all three numbers are in: the fitter
    // drops a row that is not complete, and a test that does not turn has a
    // turn of zero.
    let partial = 0;
    rows.forEach((row) => {
        const filled = ['x_mm', 'y_mm', 'turn_right_deg']
            .filter((field) => row[field] !== '' && row[field] !== null
                && row[field] !== undefined).length;
        if (filled > 0 && filled < 3) partial += 1;
    });
    let status;
    if (missing) {
        status = `${missing} of ${payload.total} measurements still needed. `
            + `Fill them in from your tape.`;
    } else {
        status = `All ${payload.total} measurements are in. Fit the model.`;
    }
    if (partial) {
        status += ` ${partial} row(s) need all three numbers: `
            + `a test that does not turn has a turn of 0.`;
    }
    el('sheet-status').textContent = status;
}

async function loadSheet() {
    try {
        const payload = await api(
            `/api/calibration/sheet?name=${encodeURIComponent(calibrationName())}`);
        renderSheet(payload);
        el('btn-fit').disabled = false;
    } catch (err) {
        el('sheet-status').textContent = err.message;
        el('sheet-table').innerHTML = '';
    }
}

async function saveSheet() {
    const rows = Array.from(el('sheet-table').querySelectorAll('tr')).slice(1).map((tr) => {
        const cells = tr.querySelectorAll('td');
        const inputs = tr.querySelectorAll('input');
        return {
            test: cells[0].textContent,
            repeat: cells[1].textContent,
            stop: cells[2].textContent,
            x_mm: inputs[0].value,
            y_mm: inputs[1].value,
            turn_right_deg: inputs[2].value,
        };
    });
    try {
        const payload = await api('/api/calibration/sheet', {
            method: 'POST',
            body: JSON.stringify({ name: calibrationName(), rows }),
        });
        renderSheet(payload);
        toast('Saved the measurement sheet.');
    } catch (err) {
        toast(err.message, true);
    }
}

async function fitModel() {
    try {
        const summary = await api('/api/calibration/fit', {
            method: 'POST',
            body: JSON.stringify({ name: calibrationName() }),
        });
        const problems = summary.problems || [];
        el('fit-status').textContent = problems.length
            ? `Fitted with warnings: ${problems.join('; ')}`
            : `Fitted. The model is saved as ${summary.path}.`;
        useModel(summary);
        await loadModels();
        toast('The measured model is ready.');
    } catch (err) {
        el('fit-status').textContent = err.message;
        toast(err.message, true);
    }
}

async function startCollect() {
    el('btn-collect').disabled = true;
    el('collect-status').textContent = 'Pushing the test suite to the hub...';
    try {
        const job = await api('/api/calibration/collect', {
            method: 'POST',
            body: JSON.stringify({
                name: calibrationName(),
                connection: state.hubConn,
                hub_name: state.hubName,
            }),
        });
        state.collectJobId = job.id;
        pollCollect();
    } catch (err) {
        el('btn-collect').disabled = false;
        el('collect-status').textContent = err.message;
        toast(err.message, true);
    }
}

function pollCollect() {
    clearTimeout(state.collectPollTimer);
    state.collectPollTimer = setTimeout(async () => {
        try {
            const job = await api(`/api/jobs/${state.collectJobId}`);
            el('log').textContent = (job.log || []).join('\n');
            const progress = job.progress || {};
            const current = progress.current;
            if (job.status === 'running') {
                el('collect-status').textContent = current
                    ? `Running ${current.test}${current.repeat !== undefined && current.repeat !== null ? `, repeat ${Number(current.repeat) + 1} of 3` : ''}. `
                        + `Press the centre button on the hub. `
                        + `${progress.needed || 0} measurements needed in total.`
                    : 'Pushing the test suite to the hub...';
                pollCollect();
                return;
            }
            el('btn-collect').disabled = false;
            if (job.status === 'ok') {
                el('collect-status').textContent = 'The session is captured. Measure the robot and fill in the sheet.';
                await loadSheet();
            } else {
                el('collect-status').textContent = job.error || 'The collection failed.';
            }
        } catch (err) {
            el('btn-collect').disabled = false;
            el('collect-status').textContent = err.message;
        }
    }, 400);
}

/* ------------------------------------------------------------- the plots */

/* The plots the command line can make, drawn from the samples the studio
 * already has. */
const PLOT_COLORS = {
    left: '#f87171',
    right: '#34d399',
    linear: '#e2e8f0',
    limit: '#fbbf24',
    convergence: '#38bdf8',
    turn: '#a78bfa',
    envelope: 'rgba(52, 211, 153, 0.9)',
};

function drawPlot() {
    const canvas = el('plot-canvas');
    if (!canvas) return;
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || 280;
    const height = 190;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const samples = state.trajectory || [];
    const limits = (state.jobResult && state.jobResult.limits) || {};
    const iterations = (state.jobResult && state.jobResult.iterations) || [];
    const times = samples.map((s) => s.t);
    const note = el('plot-note');

    const plot = PLOTS.find((entry) => entry.key === state.plot) || PLOTS[0];
    const built = plot.build({ samples, times, limits, iterations });

    if (!samples.length && plot.key !== 'convergence') {
        note.textContent = 'Solve a mission to see the plots.';
        drawPlotFrame(ctx, width, height, [], built.xlabel, built.ylabel, built.title, null);
        return;
    }
    note.textContent = built.note || '';
    drawPlotFrame(ctx, width, height, built.series, built.xlabel, built.ylabel,
        built.title, built.horizontal);
}

/* The plots. Each one says what it draws and against which limit. */
function buildSpeedPlot({ samples, times, limits }) {
    const series = [
        { label: 'Left wheel', color: PLOT_COLORS.left, x: times, y: samples.map((s) => s.vl) },
        { label: 'Right wheel', color: PLOT_COLORS.right, x: times, y: samples.map((s) => s.vr) },
        { label: 'Linear', color: PLOT_COLORS.linear, x: times, y: samples.map((s) => (s.vl + s.vr) / 2) },
    ];
    return {
        title: 'Velocity profiles',
        xlabel: 'Time (s)',
        ylabel: 'Velocity (m/s)',
        series,
        horizontal: limits.speed_limit_m_s
            ? { value: limits.speed_limit_m_s, label: 'top speed' } : null,
    };
}

function buildForcePlot({ samples, times, limits }) {
    const series = [
        { label: 'Left wheel', color: PLOT_COLORS.left, x: times, y: samples.map((s) => s.fl) },
        { label: 'Right wheel', color: PLOT_COLORS.right, x: times, y: samples.map((s) => s.fr) },
    ];
    const limit = limits.force_limit_n;
    if (limit && limit.length) {
        series.push({ label: 'Force limit', color: PLOT_COLORS.limit, x: times, y: limit });
    }
    return {
        title: 'Wheel forces',
        xlabel: 'Time (s)',
        ylabel: 'Force (N)',
        series,
    };
}

function buildAccelPlot({ samples, times }) {
    return {
        title: 'Wheel accelerations',
        xlabel: 'Time (s)',
        ylabel: 'Acceleration (m/s2)',
        series: [
            { label: 'Left wheel', color: PLOT_COLORS.left, x: times, y: samples.map((s) => s.al) },
            { label: 'Right wheel', color: PLOT_COLORS.right, x: times, y: samples.map((s) => s.ar) },
        ],
    };
}

function buildTurnPlot({ samples, times }) {
    return {
        title: 'Turn rate over time',
        xlabel: 'Time (s)',
        ylabel: 'Turn rate (rad/s)',
        series: [
            { label: 'Angular velocity', color: PLOT_COLORS.turn,
                x: times, y: samples.map((s) => s.omega) },
        ],
    };
}

/* The heading is the plan's angle in (-pi, pi], so a path that crosses the
   range would jump 180 -> -180. The plot holds one continuous turn instead,
   which is how the robot's own odometry reads it. */
function unwrapHeading(samples) {
    let offset = 0;
    let previous = null;
    return samples.map((sample) => {
        const heading = sample.heading;
        if (previous !== null) {
            const delta = heading - previous;
            if (delta > Math.PI) offset -= 2 * Math.PI;
            else if (delta < -Math.PI) offset += 2 * Math.PI;
        }
        previous = heading;
        return (heading + offset) * 180 / Math.PI;
    });
}

function buildHeadingPlot({ samples, times }) {
    return {
        title: 'Heading over time',
        xlabel: 'Time (s)',
        ylabel: 'Heading (deg)',
        series: [
            { label: 'Heading, unwrapped', color: PLOT_COLORS.convergence,
                x: times, y: unwrapHeading(samples) },
        ],
    };
}

/* The per-sample radius the gate compares with each segment's limit: half the
   diagonal of the envelope box, in cm (main.py measures the same radius). */
function buildEnvelopePlot({ samples, times }) {
    const radius = samples.map((sample) => {
        const env = sample.reachability_envelope;
        if (!env) return NaN;
        return Math.hypot((env.x_max - env.x_min) / 2, (env.y_max - env.y_min) / 2) * 100;
    });
    return {
        title: 'Reachability envelope',
        xlabel: 'Time (s)',
        ylabel: 'Envelope (cm)',
        series: [{ label: 'Half-diagonal radius', color: PLOT_COLORS.envelope,
                   x: times, y: radius }],
        note: radius.some(Number.isFinite)
            ? `The uncertainty the validator propagated. Its limit is per segment, `
              + `so the Result tab holds the verdict.`
            : 'Only a validated run has an envelope. Turn validation on in the Solve tab and solve again.',
    };
}

function buildConvergencePlot({ iterations }) {
    const costs = iterations.map((entry) => entry.cost);
    /* One phase label per run of phases: the per-iteration points all say
       "ipopt", and repeating it would bury the note. */
    const phases = iterations.reduce((list, entry) => {
        if (list[list.length - 1] !== entry.phase) list.push(entry.phase);
        return list;
    }, []);
    return {
        title: 'How the solve converged',
        xlabel: 'Step',
        ylabel: 'Cost (s)',
        series: costs.length
            ? [{ label: 'Cost', color: PLOT_COLORS.convergence,
                 x: iterations.map((entry, index) => index), y: costs,
                 points: costs.length <= 20 }]
            : [],
        note: costs.length
            ? `${phases.join(' -> ')}. `
              + `The dashed path on the field is the solver's first guess.`
            : 'The solver did not record its iterations for this run.',
    };
}

/* The plot strip, built from the registry. */
function renderPlotTabs() {
    const strip = el('plot-tabs');
    if (!strip) return;
    strip.innerHTML = '';
    PLOTS.forEach((plot) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'btn small plot-tab';
        button.dataset.plot = plot.key;
        button.textContent = plot.label;
        button.addEventListener('click', () => {
            state.plot = plot.key;
            renderPlotTabs();
            drawPlot();
        });
        strip.appendChild(button);
    });
    strip.querySelectorAll('.plot-tab').forEach((button) => {
        button.classList.toggle('active', button.dataset.plot === state.plot);
    });
}

/* The key to the field's colours, built from the same list the drawing
   uses. */
const legendRows = {};
const legendLabels = {};

/* The propagation the last verdict's envelope came from. An envelope is the
   width of a reachability tube, and the tube is a property of the model that
   produced it, so the key names the model rather than only the colour
   (KI-049). */
function envelopeModelName() {
    const report = state.validationReport;
    if (!report || !report.candidates || !report.candidates.length) return 'Plant';
    const index = state.reportCandidate !== null
        && state.reportCandidate < report.candidates.length
        ? state.reportCandidate : (report.selected || 1) - 1;
    const candidate = report.candidates[index] || report.candidates[0] || {};
    return (candidate.propagation || report.propagation) === 'tick'
        ? 'Closed-Loop Tick' : 'Plant';
}

function updateLegendLabels() {
    if (legendLabels.envelope) {
        legendLabels.envelope.textContent = `Reachability envelope (${envelopeModelName()})`;
    }
    /* The band is the field's own number, and the collision report judges
       hits by it, so the key names the value it is drawn at. */
    const field = state.fields[state.season];
    const band = field && field.placement_uncertainty_m;
    if (legendLabels.placement) {
        legendLabels.placement.textContent = band
            ? `Placement uncertainty (±${Math.round(band * 1000)} mm)`
            : 'Placement uncertainty';
    }
}

function renderLegend() {
    const box = el('legend-items');
    if (!box) return;
    box.innerHTML = '';
    for (const key of Object.keys(legendRows)) delete legendRows[key];
    for (const key of Object.keys(legendLabels)) delete legendLabels[key];
    CANVAS_LEGEND.forEach((item) => {
        const row = document.createElement('div');
        row.className = 'legend-item';
        const swatch = document.createElement('span');
        swatch.className = `legend-swatch${item.dot ? ' dot' : ''}`;
        swatch.style.background = item.dash
            ? `repeating-linear-gradient(90deg, ${item.color} 0 4px, transparent 4px 7px)`
            : item.color;
        const label = document.createElement('span');
        label.textContent = item.label;
        row.appendChild(swatch);
        row.appendChild(label);
        box.appendChild(row);
        legendRows[item.key] = row;
        legendLabels[item.key] = label;
        if (item.toggle) {
            row.dataset.toggle = item.toggle;
            row.setAttribute('role', 'button');
            row.tabIndex = 0;
            row.title = 'Click to show or hide it on the field.';
            row.addEventListener('click', () => toggleLegend(item, row));
            row.addEventListener('keydown', (event) => {
                if (event.key !== 'Enter' && event.key !== ' ') return;
                event.preventDefault();
                toggleLegend(item, row);
            });
        }
    });
    updateLegendLabels();
    updateLegend();
}

/* Where each waypoint sits in the sample list. The plan passes through the
   waypoints, so the sample nearest a waypoint is its boundary; the samples
   between two of them are one segment. */
function segmentBounds() {
    const samples = state.trajectory || [];
    const waypoints = state.waypoints || [];
    if (samples.length < 2 || waypoints.length < 2) return [];
    const bounds = [];
    let from = 0;
    waypoints.forEach((point, index) => {
        let best = from;
        let bestDistance = Infinity;
        for (let i = from; i < samples.length; i++) {
            const distance = Math.hypot(samples[i].x - point.x, samples[i].y - point.y);
            if (distance < bestDistance) {
                bestDistance = distance;
                best = i;
            }
        }
        bounds.push(index === 0 ? 0 : best);
        from = best;
    });
    bounds[bounds.length - 1] = samples.length - 1;
    return bounds;
}

/* The sample that sits at `fraction` of a segment's arc length. */
function sampleAtFraction(from, to, fraction) {
    const samples = state.trajectory;
    let total = 0;
    for (let i = from; i < to; i++) {
        total += Math.hypot(samples[i + 1].x - samples[i].x,
            samples[i + 1].y - samples[i].y);
    }
    let running = 0;
    for (let i = from; i < to; i++) {
        running += Math.hypot(samples[i + 1].x - samples[i].x,
            samples[i + 1].y - samples[i].y);
        if (running >= total * fraction) return samples[i + 1];
    }
    return samples[to];
}

/* The segments the path turns through. A segment that changes heading by
   more than 0.2 rad has an apex worth marking; a straight one does not. */
function curvedSegments() {
    const samples = state.trajectory || [];
    const bounds = segmentBounds();
    const curved = [];
    for (let segment = 0; segment + 1 < bounds.length; segment++) {
        const from = bounds[segment];
        const to = bounds[segment + 1];
        if (to - from < 2) continue;
        const turn = wrapAngle(samples[to].heading - samples[from].heading);
        if (Math.abs(turn) > 0.2) curved.push({ from, to, turn });
    }
    return curved;
}

/* The intermediate waypoints with a free heading: the ones where the robot
   carries its momentum through instead of stopping to point. */
function bisectorWaypoints() {
    return state.waypoints.filter((point, index) => index > 0
        && index < state.waypoints.length - 1
        && (point.heading === null || point.heading === undefined));
}

/* What is on the field right now. The key shows only these, so the list is
   a key to the picture and not a list of things that might appear. */
function legendVisibility() {
    const samples = state.trajectory || [];
    const field = state.fields[state.season];
    const hasHeading = state.waypoints.some(
        (point) => point.heading !== null && point.heading !== undefined);
    const hasFree = state.waypoints.some(
        (point) => point.heading === null || point.heading === undefined);
    const limit = actuatorLimit();
    const track = trackWidth();
    const saturations = limit
        ? samples.map((sample) => actuatorSaturation(sample, limit, track)) : [];
    return {
        mat: Boolean(state.mat.image),
        trajectory: samples.length > 0,
        candidates: state.candidatePaths.length > 1
            || (state.candidatePaths.length > 0 && samples.length === 0),
        envelope: samples.length > 0,
        envelopeBoxes: samples.length > 0,
        saturation: saturations.some((value) => value >= 0.85 && value <= 1.0),
        slipping: saturations.some((value) => value > 1.0),
        apex: curvedSegments().length > 0,
        bisector: bisectorWaypoints().length > 0 && samples.length > 1,
        path: state.waypoints.length > 1,
        ghost: Boolean(state.ghostStates && state.ghostStates.length),
        runs: state.runs.length > 0,
        heading: hasHeading,
        free: hasFree,
        measured: state.measuredPoses.length > 0,
        models: Boolean(field && field.obstacles && field.obstacles.length),
        placement: Boolean(field && field.obstacles && field.obstacles.length
            && field.placement_uncertainty_m > 0),
        collision: state.collisions.length > 0,
        failing: state.failingSegments.size > 0 || state.worstSegment >= 0,
    };
}

/* Only a validated run carries a reachable set, and the fast preset does not
   validate. A switch that looks live and does nothing reads as a bug, so the
   envelope entry says why it is off. */
function envelopeMissing() {
    return (envelopeSamples() || [])
        .filter((sample) => sample.reachability_envelope).length < 2;
}

function updateLegend() {
    updateLegendLabels();
    const visible = legendVisibility();
    Object.keys(legendRows).forEach((key) => {
        const row = legendRows[key];
        if (!row) return;
        row.hidden = !visible[key];
        if (!row.dataset.toggle) return;
        const on = Boolean(state[row.dataset.toggle]);
        row.setAttribute('aria-pressed', String(on));
        const blocked = (key === 'envelope' || key === 'envelopeBoxes')
            && envelopeMissing();
        if (blocked) row.setAttribute('aria-disabled', 'true');
        else row.removeAttribute('aria-disabled');
        /* The entry stays in the key while it is off - it is the switch
           back on - and the dimmed row says the field is not drawing it. */
        row.title = blocked
            ? 'No envelope yet: only a validated run has one (the fast preset does not validate).'
            : on ? 'Click to hide it on the field.' : 'Click to show it on the field.';
    });
}

/* A key entry that is also a switch: the key is where the operator looks to
   see what is on the field, so the switch belongs there too. */
function toggleLegend(item, row) {
    if (row && row.getAttribute('aria-disabled') === 'true') return;
    state[item.toggle] = !state[item.toggle];
    scheduleSessionSave();
    draw();
}

function drawPlotFrame(ctx, width, height, series, xlabel, ylabel, title, horizontal) {
    const left = 46;
    const right = width - 10;
    const bottom = height - 30;
    ctx.font = '10px system-ui, sans-serif';
    ctx.fillStyle = '#93a4bf';
    ctx.textAlign = 'left';
    ctx.fillText(title, left, 11);
    /* The legend gets its own row(s) under the title. It used to start at the
       title's baseline, so the two were printed over each other; a wrap keeps
       it inside the box when the panel is narrow and the key is long. */
    const legend = series.concat(horizontal && horizontal.label
        ? [{ label: horizontal.label, color: PLOT_COLORS.limit, dash: true }] : []);
    const legendEntries = [];
    let lx = left;
    let row = 0;
    legend.forEach((entry) => {
        const w = 13 + ctx.measureText(entry.label).width;
        if (lx > left && lx + w > right) { row += 1; lx = left; }
        legendEntries.push({ entry, x: lx, y: 25 + row * 14 });
        lx += w;
    });
    const top = 25 + row * 14 + 8;
    if (!series.length || !series.some((s) => s.x && s.x.length)) {
        return;
    }

    let xMin = Infinity;
    let xMax = -Infinity;
    let yMin = Infinity;
    let yMax = -Infinity;
    series.forEach((entry) => {
        entry.x.forEach((value, index) => {
            const y = entry.y[index];
            if (!Number.isFinite(value) || !Number.isFinite(y)) return;
            xMin = Math.min(xMin, value);
            xMax = Math.max(xMax, value);
            yMin = Math.min(yMin, y);
            yMax = Math.max(yMax, y);
        });
    });
    if (horizontal && Number.isFinite(horizontal.value)) {
        yMin = Math.min(yMin, horizontal.value);
        yMax = Math.max(yMax, horizontal.value);
    }
    if (!Number.isFinite(xMin) || !Number.isFinite(yMin)) return;
    if (xMax - xMin < 1e-9) xMax = xMin + 1;
    if (yMax - yMin < 1e-9) { yMax = yMin + 1; yMin -= 0.5; }
    const pad = (yMax - yMin) * 0.08;
    yMin -= pad;
    yMax += pad;

    const px = (value) => left + ((value - xMin) / (xMax - xMin)) * (right - left);
    const py = (value) => bottom - ((value - yMin) / (yMax - yMin)) * (bottom - top);

    // Grid and axes.
    ctx.strokeStyle = 'rgba(255,255,255,0.07)';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 3; i++) {
        const value = yMin + ((yMax - yMin) * i) / 3;
        const y = py(value);
        ctx.beginPath();
        ctx.moveTo(left, y);
        ctx.lineTo(right, y);
        ctx.stroke();
        ctx.fillText(value.toFixed(Math.abs(value) < 10 ? 2 : 1), left - 4, y + 3);
    }
    ctx.textAlign = 'center';
    for (let i = 0; i <= 3; i++) {
        const value = xMin + ((xMax - xMin) * i) / 3;
        /* The last tick would be cut by the canvas edge if it stayed centred. */
        ctx.textAlign = i === 3 ? 'right' : 'center';
        ctx.fillText(value.toFixed(2), px(value), bottom + 13);
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.18)';
    ctx.beginPath();
    ctx.moveTo(left, top);
    ctx.lineTo(left, bottom);
    ctx.lineTo(right, bottom);
    ctx.stroke();

    // A limit line.
    if (horizontal && Number.isFinite(horizontal.value)) {
        ctx.save();
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = PLOT_COLORS.limit;
        const y = py(horizontal.value);
        ctx.beginPath();
        ctx.moveTo(left, y);
        ctx.lineTo(right, y);
        ctx.stroke();
        ctx.restore();
    }

    // The series.
    series.forEach((entry) => {
        ctx.strokeStyle = entry.color;
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        let started = false;
        entry.x.forEach((value, index) => {
            const y = entry.y[index];
            if (!Number.isFinite(value) || !Number.isFinite(y)) return;
            if (!started) { ctx.moveTo(px(value), py(y)); started = true; }
            else ctx.lineTo(px(value), py(y));
        });
        ctx.stroke();
        if (entry.points) {
            entry.x.forEach((value, index) => {
                const y = entry.y[index];
                if (!Number.isFinite(value) || !Number.isFinite(y)) return;
                ctx.fillStyle = entry.color;
                ctx.beginPath();
                ctx.arc(px(value), py(y), 3, 0, Math.PI * 2);
                ctx.fill();
            });
        }
    });

    // The legend.
    ctx.textAlign = 'left';
    legendEntries.forEach(({ entry, x, y }) => {
        ctx.strokeStyle = entry.color;
        ctx.setLineDash(entry.dash ? [3, 3] : []);
        ctx.beginPath();
        ctx.moveTo(x, y - 3);
        ctx.lineTo(x + 8, y - 3);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#93a4bf';
        ctx.fillText(entry.label, x + 11, y);
    });

    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'right';
    ctx.fillText(ylabel, right, height - 4);
    ctx.textAlign = 'center';
    ctx.fillText(xlabel, (left + right) / 2, height - 4);
}

/* The solver's first guess, drawn on the field next to the answer. */
function drawGhostPath() {
    const ghost = state.ghostStates;
    if (!ghost || ghost.length < 2) return;
    const ratio = window.devicePixelRatio || 1;
    ctx.save();
    ctx.setLineDash([5 * ratio, 5 * ratio]);
    ctx.strokeStyle = 'rgba(226, 232, 240, 0.35)';
    ctx.lineWidth = 1.5 * ratio;
    ctx.beginPath();
    ghost.forEach((state, index) => {
        const p = worldToScreen(state[0], state[1]);
        if (index === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();
    ctx.restore();
}

/* ------------------------------------------------- what to try next */

/* When a validation fails, name the lever the measurements point at and
 * offer to pull it: the tube's model for an envelope failure, the gradient
 * polish for a tracking error just over the gate, the soft force weight for
 * a force or slip failure, and the accuracy weight for a rough profile. The
 * timing model is not a lever any more - the planner uses the per-segment
 * model for every candidate. */
function suggestFixes(report) {
    const fixes = [];
    if (!report || !report.candidates || !report.candidates.length) return fixes;
    const candidates = report.candidates;
    const selected = candidates[(report.selected || 1) - 1] || candidates[0];
    if (selected && selected.ok) return fixes;

    /* The plant tube does not see the feedback that contracts it. The
       failure is read from the per-segment excess over the active local
       limit, not from a hard-coded 2 cm budget. */
    const envelopeFailed = candidates.some((entry) => !entry.ok && (
        entry.envelope_truncated
        || (entry.segments || []).some((segment) => segment && segment.ok === false
            && (Number.isFinite(segment.excess_m) && segment.excess_m > 0
                || (Number.isFinite(segment.envelope_m) && Number.isFinite(segment.limit_m)
                    && segment.envelope_m > segment.limit_m)))));
    if (envelopeFailed && el('opt-propagation').value !== 'tick') {
        fixes.push({ id: 'tick', label: 'Try Closed-Loop Tick Propagation',
                     propagation: 'tick' });
    }

    /* A force or slip failure is what the soft force weight prices. */
    const forceFailed = candidates.some((entry) => !entry.ok
        && (entry.reasons || []).some((reason) => /force|slip|traction/i.test(reason)));
    if (forceFailed && !(Number(el('opt-force-weight').value) > 0)) {
        fixes.push({ id: 'force', label: 'Add Force Weight (0.005)', force_weight: 0.005 });
    }

    // The tracking gate with no weight: a small weight smooths the profile.
    const tracking = candidates.some((entry) => (entry.reasons || [])
        .some((reason) => reason.indexOf('tracking') >= 0));
    const weight = Number(el('opt-accuracy').value);
    if (tracking && (!weight || weight > 0.001)) {
        fixes.push({ id: 'weight', label: 'a weight of 0.001', weight: 0.001 });
    }

    return fixes;
}

/* Pull the lever the button names, then solve again. The next result is
   compared with the previous one, so the lever history is measured. */
function applyFix(fix) {
    if (fix.weight !== undefined) el('opt-accuracy').value = fix.weight;
    if (fix.propagation !== undefined) el('opt-propagation').value = fix.propagation;
    if (fix.force_weight !== undefined) el('opt-force-weight').value = fix.force_weight;
    state.pendingLever = { id: fix.id, label: fix.label };
    syncPresetRadio();
    scheduleSessionSave();
    solve();
}

/* A passing candidate that is not the one the rule exported can be exported
   instead: the operator has seen the three paths and the scores, which is
   the point of the comparison. The pick is a solve option, so it means one
   more run with that candidate pinned - and the CLI refuses it if it does
   not pass again, so the export gate stays the CLI's. */
function renderPick(report, index) {
    const row = el('pick-row');
    row.innerHTML = '';
    const candidate = report.candidates[index];
    const exported = (report.selected || 1) - 1;
    const rescue = Boolean(candidate && candidate.rescue)
        || isRescueCandidate((candidate && candidate.style) || 'baseline');
    const pickable = candidate && candidate.ok && index !== exported && !rescue;
    row.hidden = !pickable;
    if (!pickable) return;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'btn small';
    button.textContent = `Solve again and export: ${candidateName(candidate.style, rescue)}`;
    button.title = 'The plan is solved again with this candidate pinned, and exported only '
        + 'if it still passes the validator.';
    button.addEventListener('click', () => {
        state.pickCandidate = index + 1;
        solve();
    });
    row.appendChild(button);
}

function renderFixes(report) {
    const row = el('fix-row');
    row.innerHTML = '';
    const fixes = suggestFixes(report);
    row.hidden = fixes.length === 0;
    fixes.forEach((fix) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'btn small fix-btn';
        button.textContent = fix.label;
        button.addEventListener('click', () => applyFix(fix));
        row.appendChild(button);
    });
}

/* --------------------------------------------------------- nudging */

/* Move the selected waypoint with the arrow keys. A held key is one undo
 * step, not one per repeat. */
const NUDGE_STEP = 0.01;      // 1 cm
let lastNudge = { index: -1, time: 0 };

function nudgeSelected(dx, dy) {
    const point = currentWaypoint();
    if (!point) return;
    const now = Date.now();
    const continuing = lastNudge.index === state.selected && (now - lastNudge.time) < 700;
    point.x = Math.round((point.x + dx) * 1000) / 1000;
    point.y = Math.round((point.y + dy) * 1000) / 1000;
    markMissionChanged();
    if (!continuing) {
        // One undo step per gesture: the state before the nudge is still in
        // the history, so the new state is what gets pushed.
        pushHistory();
    }
    lastNudge = { index: state.selected, time: now };
    renderEditor();
    updatePathSummary();
    updateMissionState();
    draw();
    scheduleSessionSave();
}

/* --------------------------------------------- how this solve compares */

/* The faint paths on the field show the earlier solves; this is the number. */
function describeSolveDiff(previousEntry, currentEntry) {
    if (!previousEntry || !currentEntry) return '';
    const previous = previousEntry.summary;
    const current = currentEntry.summary;
    if (!previous || !current) return '';
    const parts = [];
    const delta = current.duration_s - previous.duration_s;
    if (Math.abs(delta) >= 0.005) {
        parts.push(`${Math.abs(delta).toFixed(2)} s ${delta < 0 ? 'faster' : 'slower'}`);
    }
    if (current.has_envelope && previous.has_envelope) {
        const width = (current.max_envelope_m - previous.max_envelope_m) * 1000;
        if (Math.abs(width) >= 0.5) {
            parts.push(`the envelope ${Math.abs(width).toFixed(1)} mm ${width < 0 ? 'tighter' : 'wider'}`);
        }
    }
    const length = (current.path_length_m - previous.path_length_m) * 1000;
    if (Math.abs(length) >= 1) {
        parts.push(`the path ${Math.abs(length).toFixed(0)} mm ${length < 0 ? 'shorter' : 'longer'}`);
    }
    if (previousEntry.verdict !== currentEntry.verdict) {
        parts.push(`and it now ${currentEntry.verdict === 'ok' ? 'passes' : 'fails'}`);
    }
    if (!parts.length) return 'The same as the previous solve.';
    return `vs. the previous solve: ${parts.join(', ')}.`;
}

/* The run's verdict, separate from the raw validator verdict. Acceptance is
   a decision against a threshold; it never turns a failed plan into a
   passed one, and the score is not a probability. */
function runVerdict(report, selected, jobFailed) {
    const candidates = report && report.candidates;
    const entry = selected || {};
    const rawOk = Boolean(entry.ok);
    const accepted = Boolean(report && (report.accepted || entry.accepted));
    const bestEffort = Boolean(report && (report.best_effort || entry.best_effort));
    const score = entry.score !== undefined ? entry.score
        : (report && report.score !== undefined ? report.score : null);
    const out = {
        kind: 'unvalidated',
        rawOk, accepted, bestEffort,
        score,
        scoreVersion: entry.score_version || (report && report.score_version) || 1,
        profile: entry.profile || (report && report.profile) || 'balanced',
        categories: entry.categories || (report && report.categories) || null,
        /* The field-clearance channels (--field-clearance), for the Physical
           category: the worst util (1.0 = touching) and the worst
           penetration, so a physical failure names its number. */
        clearance: {
            util: entry.field_clearance_util !== null
                && entry.field_clearance_util !== undefined
                ? entry.field_clearance_util
                : (report ? report.field_clearance_util : null),
            penetration_m: entry.penetration_m !== null
                && entry.penetration_m !== undefined
                ? entry.penetration_m
                : (report ? report.penetration_m : null),
        },
        hubApproved: Boolean(report && (report.hub_approved !== undefined
            ? report.hub_approved : (rawOk && !jobFailed))),
    };
    if (jobFailed) out.kind = 'structural';
    else if (report && candidates) {
        if (rawOk) out.kind = 'passed';
        else if (accepted) out.kind = 'accepted';
        else if (bestEffort) out.kind = 'best_effort';
        else out.kind = 'failed';
    }
    return out;
}

/* Measured levers only: the delta is taken from the two finished runs, so
   there is never a predicted points/second before a solve. */
function recordLever(previous, entry, wallS) {
    if (!state.pendingLever || !previous || !entry) return;
    const before = Number.isFinite(previous.score) ? previous.score : null;
    const after = Number.isFinite(entry.score) ? entry.score : null;
    const deltaScore = before !== null && after !== null ? after - before : null;
    const deltaDuration = Number.isFinite(previous.summary && previous.summary.duration_s)
            && Number.isFinite(entry.summary && entry.summary.duration_s)
        ? entry.summary.duration_s - previous.summary.duration_s : null;
    const wall = Number.isFinite(wallS) ? wallS : entry.wallS;
    state.leverHistory.push({
        id: state.pendingLever.id,
        label: state.pendingLever.label,
        deltaScore, deltaDuration,
        wallS: wall,
        pointsPerS: deltaScore !== null && Number.isFinite(wall) && wall > 0
            ? deltaScore / wall : null,
    });
    state.pendingLever = null;
}

function renderScoreBlock(verdict) {
    const box = el('score-block');
    const value = el('score-value');
    const meta = el('score-meta');
    const categories = el('score-categories');
    const note = el('score-note');
    if (!verdict || verdict.score === null || verdict.score === undefined) {
        box.hidden = true;
        value.textContent = '';
        meta.textContent = '';
        categories.innerHTML = '';
        note.textContent = '';
        return;
    }
    box.hidden = false;
    value.textContent = `${Number(verdict.score).toFixed(1)} / 100`;
    meta.textContent = `Decision score v${verdict.scoreVersion}, profile ${verdict.profile}. `
        + 'This is a decision score, not a probability of hardware success.';
    categories.innerHTML = '';
    const names = [['physical', 'Physical'], ['accuracy', 'Accuracy'],
        ['quality', 'Motion Quality'], ['duration', 'Duration']];
    const parts = verdict.categories || {};
    const clearance = verdict.clearance || {};
    names.forEach(([key, label]) => {
        const item = document.createElement('div');
        item.className = 'score-category';
        const number = parts[key];
        let text = `${label}: ${Number.isFinite(Number(number))
            ? Number(number).toFixed(1) : '-'}`;
        if (key === 'physical' && Number.isFinite(Number(clearance.util))) {
            /* The field-clearance channel under Physical: the util is the
               tube over the unclamped clearance (1.0 = touching), and the
               penetration is how far past it the tube reaches. */
            const past = Number(clearance.penetration_m);
            text += ` (field clearance ${Number(clearance.util).toFixed(2)}x`
                + `${Number.isFinite(past) && past > 0
                    ? `, ${(past * 100).toFixed(2)} cm past` : ''})`;
        }
        item.textContent = text;
        categories.appendChild(item);
    });
    note.textContent = verdict.kind === 'accepted'
        ? 'The plan failed validation and was accepted at this score. It is not a passed plan.'
        : verdict.kind === 'passed'
            ? 'The validator passed this plan.'
            : 'The plan is on the field for inspection.';
}

function renderLeverHistory() {
    const box = el('lever-history');
    const table = el('lever-table');
    const note = el('lever-note');
    if (!state.leverHistory.length) {
        box.hidden = true;
        table.innerHTML = '';
        note.textContent = '';
        return;
    }
    box.hidden = false;
    table.innerHTML = '<tr><th>Lever</th><th>Score</th><th>Time</th><th>Measured</th></tr>';
    const ranked = state.leverHistory.slice().sort((a, b) => {
        const left = Number.isFinite(a.deltaScore) ? a.deltaScore : -Infinity;
        const right = Number.isFinite(b.deltaScore) ? b.deltaScore : -Infinity;
        return right - left;
    });
    ranked.forEach((lever) => {
        const row = document.createElement('tr');
        const score = Number.isFinite(lever.deltaScore)
            ? `${lever.deltaScore >= 0 ? '+' : ''}${lever.deltaScore.toFixed(2)}` : '-';
        const duration = Number.isFinite(lever.deltaDuration)
            ? `${lever.deltaDuration >= 0 ? '+' : ''}${lever.deltaDuration.toFixed(2)} s` : '-';
        const measured = Number.isFinite(lever.pointsPerS)
            ? `${lever.pointsPerS.toFixed(2)} points/s` : 'wall '
                + (Number.isFinite(lever.wallS) ? `${lever.wallS.toFixed(1)} s` : '-');
        row.innerHTML = `<td>${lever.label}</td><td>${score}</td>`
            + `<td>${duration}</td><td>${measured}</td>`;
        table.appendChild(row);
    });
    note.textContent = 'Ranked by measured score improvement. Every number here is '
        + 'measured after the solve; there are no predicted estimates.';
}

function recordSolve(summary, verdict) {
    const entry = { summary, verdict, at: Date.now(), score: state.jobScore,
        wallS: state.jobWallS, outcome: state.runOutcome };
    const previous = state.solveHistory[state.solveHistory.length - 1];
    recordLever(previous, entry, state.jobWallS);
    state.solveHistory.push(entry);
    if (state.solveHistory.length > 10) state.solveHistory.shift();
    el('solve-diff').textContent = describeSolveDiff(previous, entry);
    renderLeverHistory();
}

/* ------------------------------------------------------------- rendering */

function renderAll() {
    renderWaypointList();
    renderEditor();
    updateWaypointBadge();
    renderResult();
    updateWaypointCount();
    updateMissionState();
    renderRobotChecks();
}

function updateWaypointCount() {
    const count = state.waypoints.length;
    el('waypoint-count').textContent = count === 0
        ? 'No waypoints yet'
        : `${count} waypoint${count === 1 ? '' : 's'}`;
    updatePathSummary();
}

/* The length of the path through the waypoints, and the fastest the robot
 * could possibly do it. Both are known before the solve, so a waypoint that
 * was dragged 5 m away shows up here instead of in the solve time. */
function pathLength() {
    let total = 0;
    for (let i = 0; i < state.waypoints.length - 1; i++) {
        const a = state.waypoints[i];
        const b = state.waypoints[i + 1];
        total += Math.hypot(b.x - a.x, b.y - a.y);
    }
    return total;
}

function topSpeed() {
    const robot = robotFields();
    const vMax = Number(robot.v_max_rad_s);
    const radius = Number(robot.wheel_radius);
    if (!Number.isFinite(vMax) || !Number.isFinite(radius) || vMax <= 0 || radius <= 0) {
        return null;
    }
    const headroom = Number(robot.speed_headroom);
    const factor = Number.isFinite(headroom) && headroom > 0 ? headroom : 1;
    return vMax * radius * factor;
}

function updatePathSummary() {
    const node = el('path-summary');
    if (state.waypoints.length < 2) {
        node.textContent = 'Add waypoints to see the path length.';
        return;
    }
    const length = pathLength();
    const speed = topSpeed();
    if (!speed) {
        node.textContent = `Path ${length.toFixed(2)} m`;
        return;
    }
    const fastest = length / speed;
    node.textContent = `Path ${length.toFixed(2)} m | at least ${fastest.toFixed(2)} s `
        + `at ${speed.toFixed(2)} m/s top speed`;
}

function updateMissionState() {
    const node = el('mission-state');
    if (!state.savedName) {
        node.textContent = 'Not saved to missions/ yet';
        node.className = 'hint';
    } else if (state.savedFingerprint !== missionFingerprint(false)) {
        /* The file on disk holds the waypoints, so the waypoints are what
           says whether the screen still shows what was saved. */
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
        // Built from text nodes, not from an HTML string: the event name is
        // free text from the editor or from a mission file, and a file can
        // carry anything.
        const label = document.createElement('span');
        label.appendChild(document.createTextNode(`W${index} `));
        const coords = document.createElement('span');
        coords.className = 'mono';
        coords.textContent = `(${point.x.toFixed(2)}, ${point.y.toFixed(2)})`;
        label.appendChild(coords);
        const tagBox = document.createElement('span');
        tagBox.className = 'tags';
        tagBox.textContent = tags.join(', ');
        item.appendChild(label);
        item.appendChild(tagBox);
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
    updateWaypointBadge();
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

/* The Waypoint tab carries a mark when something is selected, so the
   editor is never a hidden panel with something to say. */
function updateWaypointBadge() {
    const badge = el('tab-waypoint-badge');
    if (!badge) return;
    const has = state.selected >= 0 && state.selected < state.waypoints.length;
    badge.hidden = !has;
    badge.textContent = has ? `W${state.selected}` : '';
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
    if (state.useModel && state.model) {
        const gain = state.model.straight && state.model.straight.gain;
        rows.push(['Model', gain
            ? `measured (straight gain ${gain.toFixed(3)})`
            : 'measured']);
    }
    if (state.jobWallS !== null && state.jobWallS !== undefined) {
        rows.push(['Solve (wall)', `${state.jobWallS.toFixed(2)} s`]);
        const speed = topSpeed();
        if (speed && summary.path_length_m > 0) {
            /* The ideal is a straight run at the top speed. The plan cannot
               beat it, so the number says how much of it it got. */
            const fastest = summary.path_length_m / speed;
            rows.push(['Time vs. top speed',
                `${(summary.duration_s / fastest).toFixed(2)}x the ideal (${fastest.toFixed(2)} s)`]);
        }
    }
    renderCollisions(rows);
    rows.forEach(([key, value]) => {
        const dt = document.createElement('dt');
        dt.textContent = key;
        const dd = document.createElement('dd');
        dd.textContent = value;
        facts.appendChild(dt);
        facts.appendChild(dd);
    });
}

/* What the planned path ran into, and how sure we are.

   A collision shallower than the field's placement uncertainty is not a
   hit: the footprint's own position on the mat is not known to better than
   that, so it is marked as uncertain rather than reported as a fact. */
function renderCollisions(rows) {
    const box = el('collisions');
    if (!box) return;
    box.innerHTML = '';
    if (!state.collisions.length) {
        if (state.season) {
            rows.push(['Field', `no collisions on ${state.season}`]);
        }
        return;
    }
    const solid = state.collisions.filter((item) => !item.uncertain).length;
    const unsure = state.collisions.length - solid;
    let text = `${solid} collision${solid === 1 ? '' : 's'}`;
    if (unsure) {
        const field = state.fields[state.season];
        const band = field && field.placement_uncertainty_m;
        text += `, ${unsure} within the placement uncertainty`
            + (band ? ` (±${Math.round(band * 1000)} mm)` : '');
    }
    rows.push(['Field', text]);

    const list = document.createElement('ul');
    list.className = 'collision-list';
    state.collisions.forEach((item) => {
        const entry = document.createElement('li');
        entry.className = item.uncertain ? 'warn' : 'error';
        entry.textContent = `${item.name} at ${item.t.toFixed(2)} s, `
            + `${(item.depth_m * 1000).toFixed(0)} mm deep`
            + (item.uncertain ? ' (within the placement uncertainty)' : '');
        list.appendChild(entry);
    });
    box.appendChild(list);
}

function renderDownloads(artifacts, jobId) {
    const box = el('downloads');
    box.innerHTML = '';
    const labels = {
        traj: 'Trajectory (.traj)',
        controller: 'Controller (.json)',
        python: 'Python module',
        stats: 'Stats (.json)',
    };
    /* The job the artifacts belong to, not the current one: after a push
       the downloads box holds the push job's artifacts, and asking the
       solve job for them is a 404. */
    const owner = jobId || state.jobId;
    (artifacts || []).forEach((artifact) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'btn small';
        button.textContent = labels[artifact.kind] || artifact.kind;
        button.addEventListener('click', () => {
            window.location.href = `/api/jobs/${owner}/artifact/${artifact.kind}`;
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
            // The history entry is pushed on the commit, so a field that is
            // typed into is one undo step, not one per keystroke.
            const updated = robotFields();
            if (input.value === '') {
                delete updated[spec.key];
            } else {
                updated[spec.key] = value / (spec.scale || 1);
            }
            setRobotFields(updated);
            markMissionChanged();
            renderRobotFacts();
            renderRobotChecks();
            updateMissionState();
            updatePathSummary();
            scheduleSessionSave();
        });
        // The undo step is taken when the field is committed, so a field that is
        // typed into is one step, not one per keystroke.
        input.addEventListener('change', () => pushHistory());
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
                markMissionChanged();
                scheduleSessionSave();
            });
            input.addEventListener('change', () => pushHistory());
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
    const order = ['fast', 'safe'];
    /* What each card actually runs. */
    const descriptions = {
        fast: 'One candidate with the default settings, no validation. Seconds.',
        safe: 'Three candidates compared and validated: minutes, not seconds - the '
            + 'reachability check is the cost, so pick this only to ship.',
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
    // A preset's sample count only reaches the solver when the automatic mesh
    // is off (gui_server passes -n in exactly that case), and the CLI does the
    // same for --preset, so the tick goes with it.
    el('opt-auto-mesh').checked = false;
    el('opt-accuracy').value = preset.accuracy_weight;
    el('opt-candidates').value = preset.candidates;
    el('opt-validate').checked = Boolean(preset.validate);
    el('opt-export').value = preset.export_format;
    el('opt-propagation').value = preset.propagation === 'tick' ? 'tick' : 'plant';
    document.querySelectorAll('#presets input[type=radio]').forEach((radio) => {
        radio.checked = radio.value === name;
    });
    updateGateFields();
    updateMeshField();
    scheduleSessionSave();
}

/* The gate owns its block: every field under "Checked" is only sent with
   --validate (gui_server refuses to pass them on their own), so off the gate
   they are dead controls. A control that looks live and does nothing reads
   as a bug. */
function updateGateFields() {
    const on = el('opt-validate').checked;
    const fields = {
        'opt-candidates': ['Every candidate is solved and checked.',
            'Candidates are only compared when validation is on.'],
        'opt-propagation': ['The model the envelope gate uses.',
            'The propagation model is only used by validation.'],
        'opt-envelope-limit': ['How far the reachable set may spread around the plan. '
            + 'A waypoint error threshold still wins.',
            'The accuracy limit is only applied by validation.'],
        'opt-envelope-limit-per-metre': ['How much the limit may grow per metre travelled.',
            'The allowance is only applied by validation.'],
        'opt-best-effort': ['A failed validation still writes the best path, marked failed.',
            'Only a validation can fail, so there is nothing to keep.'],
        'opt-score-profile': ['The weights the decision score uses.',
            'The score is built from a validation.'],
        'opt-target-time': ['The duration the score prices.',
            'The target time is only used by validation.'],
        'opt-max-time': ['The duration the score will not exceed.',
            'The maximum time is only used by validation.'],
        'opt-jerk-target': ['The wheel-jerk the score prices.',
            'The jerk target is only used by validation.'],
        'opt-jerk-max': ['The wheel-jerk the score will not exceed.',
            'The jerk maximum is only used by validation.'],
        'opt-accept-score': ['A failed plan is accepted at or above this score.',
            'Acceptance only exists against a validation.'],
    };
    Object.keys(fields).forEach((id) => {
        const control = el(id);
        control.disabled = !on;
        control.title = fields[id][on ? 0 : 1];
    });
}

/* The manual samples field only means something when the automatic mesh is
   off: with it on, the 3-Pillar engine sizes the invariant mesh from the
   path and the number in the field is not used. */
function updateMeshField() {
    const auto = el('opt-auto-mesh');
    const samples = el('opt-samples');
    if (!auto || !samples) return;
    samples.disabled = auto.checked;
    samples.title = auto.checked
        ? 'The 3-Pillar engine sizes the invariant 50 ms mesh from the path.'
        : 'Samples per segment on every segment.';
}

/* ----------------------------------------------------------------- solve */

function optionalNumber(id) {
    const raw = el(id).value;
    return raw === '' ? null : Number(raw);
}

function collectOptions() {
    return {
        samples: Number(el('opt-samples').value),
        auto_mesh: el('opt-auto-mesh').checked,
        accuracy_weight: Number(el('opt-accuracy').value),
        candidates: Number(el('opt-candidates').value),
        force_weight: Number(el('opt-force-weight').value),
        propagation: el('opt-propagation').value,
        validate: el('opt-validate').checked,
        /* The page asks in cm, the command line takes metres. */
        envelope_limit: Number(el('opt-envelope-limit').value) / 100,
        /* What the tube may grow per metre travelled: 0 keeps the limit
           absolute. The server only forwards this with --validate. */
        envelope_limit_per_metre: Number(el('opt-envelope-limit-per-metre').value || 0) / 100,
        /* A failed validation still has a plan worth looking at: the best
           candidate is exported and marked failed instead of leaving the
           field empty. The server only forwards this with --validate, and
           the push stays gated on the verdict. */
        best_effort: el('opt-best-effort').checked,
        fast_polish: el('opt-fast-polish').checked,
        scoring_profile: el('opt-score-profile').value,
        target_time: optionalNumber('opt-target-time'),
        max_time: optionalNumber('opt-max-time'),
        jerk_target: optionalNumber('opt-jerk-target'),
        jerk_max: optionalNumber('opt-jerk-max'),
        accept_score: el('opt-validate').checked ? optionalNumber('opt-accept-score') : null,
        iteration_history: true,
        sysid_fit: (el('opt-use-model').checked && state.model) ? state.model.path : null,
        export_format: el('opt-export').value,
        controller_dt: Number(el('opt-controller-dt').value),
        field: state.season,
    };
}

/* The message the operator reads, without the internal details. The CLI
   names the file it would have written - a temporary folder that is gone
   once the job ends - and the solver log keeps the full text. */
function cleanError(message) {
    if (!message) return '';
    return String(message)
        .replace(/^Error:\s*/i, '')
        .replace(/No trajectory written to '[^']*'\.?/g, 'No trajectory was written.')
        .trim();
}

/* A solve that never starts still has to clear the last verdict. Otherwise
   the header keeps saying "Failed" from a solve that has nothing to do with
   the mission on the field. */
function refuseSolve(message) {
    setStatus(message, 'warn');
    setResultStatus('warn', 'Not solved', message);
    toast(message, true);
}

/* The path on the field is the path of the last solve. When there is no
   solve - none yet, or one that failed - the field shows the waypoints and
   nothing else, so a stale path cannot be mistaken for the answer. */
function clearTrajectory() {
    state.trajectory = null;
    playback.duration = 0;
    playback.time = 0;
    playback.playing = false;
    el('playback').value = 0;
    el('btn-play').textContent = 'Play';
    updatePlaybackLabel();
}

async function solve() {
    if (state.waypoints.length < 2) {
        refuseSolve('Add at least two waypoints before solving.');
        return;
    }
    if (!state.config) {
        refuseSolve('Choose a robot configuration first.');
        return;
    }
    const options = collectOptions();
    if (options.validate) {
        if (!Number.isFinite(options.target_time) || !Number.isFinite(options.max_time)
                || !(options.target_time > 0) || !(options.target_time < options.max_time)) {
            refuseSolve('A validated solve needs a target time above zero and a maximum time above it.');
            return;
        }
    }
    setBusy(true);
    setStatus('Solving...');
    el('result-status').textContent = 'Solving...';
    el('validate-row').hidden = true;
    el('log').textContent = '';
    state.jobSummary = null;
    state.jobWallS = null;
    state.collisions = [];
    state.candidatePaths = [];
    state.pollRetries = 0;
    state.solveJobId = null;
    state.solveHasTrajectory = false;
    state.solveValidated = false;
    state.measuredPoses = [];
    /* The last run's verdict goes with the last run. A solve without
       validation prints no report, so nothing would take the old one off
       the screen and the new run would be read through it. */
    renderValidationReport(null);
    /* The previous path is dropped here, not when the new one arrives. A
       solve that fails must not leave the last solve's path on the field
       looking like the answer to this one; it is remembered as an earlier
       run instead. */
    clearTrajectory();
    updatePushButton();
    renderResult();
    renderDownloads([]);
    /* The mission that is going to be solved, so the answer can be checked
       against the mission on the screen when it comes back. */
    state.solvingFingerprint = missionFingerprint();
    /* The operator's pick from the last comparison, when there is one: the
       run exports that candidate instead of the fastest passing one. It is
       consumed here, so a later solve is not quietly pinned to the same
       seed; the CLI refuses the pick if that candidate no longer passes. */
    if (state.pickCandidate) {
        options.pick_candidate = state.pickCandidate;
        options.candidates = Math.max(Number(options.candidates) || 1,
            state.pickCandidate);
        state.pickCandidate = null;
    }
    try {
        const job = await api('/api/solve', {
            method: 'POST',
            body: JSON.stringify({
                name: el('mission-name').value,
                config: state.config,
                mission: missionDocument(),
                waypoints: state.waypoints.map(waypointToJson),
                options,
            }),
        });
        state.jobId = job.id;
        pollJob();
    } catch (err) {
        setBusy(false);
        setStatus(err.message, 'error');
        setResultStatus('error', 'Failed', err.message);
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
                /* A validated run writes each path as it is solved, before
                   its check and before the rescue round: draw them so the
                   field is not empty while the minutes pass. */
                const paths = (job.result && job.result.candidates) || [];
                if (paths.length !== state.candidatePaths.length) {
                    state.candidatePaths = paths;
                    draw();
                }
                setStatus(solveProgressText(job));
                pollJob();
                return;
            }
            setBusy(false);
            /* Every candidate the run solved, paths included. A failed
               validation still wrote them, and that is exactly when the
               comparison is worth looking at. */
            state.candidatePaths = (job.result && job.result.candidates) || [];
            if (job.status === 'ok') {
                state.jobSummary = job.summary;
                state.jobWallS = job.wall_s;
                state.trajectory = await api(`/api/jobs/${state.jobId}/trajectory`);
                state.trajectory = state.trajectory.trajectory.samples;
                rememberRun(state.trajectory);
                playback.duration = state.trajectory.length
                    ? state.trajectory[state.trajectory.length - 1].t : 0;
                playback.time = 0;
                el('playback').value = 0;
                state.solveJobId = state.jobId;
                state.solveHasTrajectory = (job.artifacts || [])
                    .some((artifact) => artifact.kind === 'traj');
                /* The validator's own verdict, not the switch on the Solve
                   tab: a solve with a measured model is validated even when
                   the box was not ticked. */
                const selected = report && report.candidates
                    ? (report.candidates[(report.selected || 1) - 1] || {}) : {};
                state.solveValidated = Boolean(selected.ok);
                const verdict = runVerdict(report, selected, false);
                state.runOutcome = verdict.kind;
                state.hubApproved = verdict.hubApproved;
                state.jobScore = verdict.score;
                renderScoreBlock(verdict);
                /* A best-effort run is a successful job whose verdict is a
                   failure: the path is on the field to be inspected, not to
                   be approved. Say which of the two it is. */
                const failedValidation = Boolean(report && selected
                    && selected.ok === false);
                /* The trajectory is the answer to the mission that is on the
                   screen, so the mission has not moved on since the solve
                   and the trajectory can be pushed. When the mission was
                   changed while the solve ran, it is not, and the studio
                   says so instead of offering the old path. */
                if (state.solvingFingerprint
                        && missionFingerprint() === state.solvingFingerprint) {
                    state.dirty = false;
                }
                updatePushButton();
                state.jobResult = job.result || {};
                state.collisions = state.jobResult.collisions || [];
                const iterations = state.jobResult.iterations || [];
                const guess = iterations.find((entry) => entry.phase === 'initial_guess');
                state.ghostStates = guess ? guess.trajectory : null;
                recordSolve(job.summary, 'ok');
                state.jobFailed = false;
                state.jobFailedReason = '';
                state.jobFinishedAt = new Date().toISOString();
                renderResult();
                renderDownloads(job.artifacts, job.id);
                drawPlot();
                const validated = state.solveValidated;
                el('result-status').innerHTML = verdict.kind === 'passed'
                    ? '<span class="badge ok">Passed</span> The validator passed this plan and it is ready to use.'
                    : verdict.kind === 'accepted'
                        ? '<span class="badge warn">Accepted plan</span> This plan failed validation '
                            + `and was accepted at score ${Number(verdict.score).toFixed(1)}. `
                            + 'It is not a passed plan.'
                        : verdict.kind === 'best_effort'
                            ? '<span class="badge error">Failed validation</span> The best path is '
                                + 'shown for inspection: ' + bestEffortDetail(selected)
                                + ' It is not approved for the hub.'
                            : '<span class="badge warn">Not validated</span> This trajectory has not been '
                                + 'checked against your robot. Validate it before you drive it.';
                /* One click for the lever the message names. A failed verdict
                   has already been checked, so the lever is not "check it
                   again"; the validation detail carries the levers that fit. */
                el('validate-row').hidden = validated || failedValidation;
                // The answer goes in the header, so it is on screen whatever
                // the column is showing.
                const headline = resultHeadline();
                setStatus(headline
                    ? `${headline} - ${validated ? 'validated'
                        : (failedValidation ? 'validation failed' : 'not validated')}`
                    : 'Done', failedValidation ? 'warn' : 'ok');
                showPanel('panel-result');
            } else if (job.status === 'cancelled') {
                el('result-status').innerHTML = '<span class="badge warn">Cancelled</span>';
                setStatus('Cancelled', 'warn');
                el('validate-row').hidden = true;
                clearTrajectory();
                updatePushButton();
            } else {
                const reason = cleanError(job.error || 'The solve failed.');
                state.jobFailed = true;
                state.jobFailedReason = reason;
                state.jobFinishedAt = new Date().toISOString();
                setResultStatus('error', 'Failed', reason);
                setStatus('Failed', 'error');
                el('validate-row').hidden = true;
                /* The field must not keep showing the last solve's path as
                   if it were the answer to this one. */
                clearTrajectory();
                updatePushButton();
                // A failure is the one thing that must not be hidden behind a
                // tab, so the result comes forward.
                showPanel('panel-result');
                toast(reason, true);
            }
            draw();
        } catch (err) {
            // A blip (a reload, a closed connection) should not kill the poll.
            if (state.jobId && state.pollRetries < 3) {
                state.pollRetries += 1;
                pollJob();
                return;
            }
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

/* What the solver is doing right now: the elapsed time, the candidate it is
 * on, whether the candidates run in parallel, and whether it is in the
 * reachability check - which is where the time goes on a long mission, not
 * the solve. */
function solveProgressText(job) {
    const log = job.log || [];
    let stage = '';
    let candidate = '';
    let candidateAt = -1;
    let checkAt = -1;
    for (let i = log.length - 1; i >= 0; i--) {
        const line = log[i];
        if (candidateAt < 0) {
            const match = line.match(/(Candidate|Rescue) (\d+)\/(\d+)/);
            if (match) {
                candidate = `${match[1].toLowerCase()} ${match[2]}/${match[3]}`;
                candidateAt = i;
            }
        }
        if (!stage) {
            const parallel = line.match(/Solving (\d+) candidates \(in parallel\)/);
            if (parallel) stage = `${parallel[1]} candidates in parallel`;
        }
        if (checkAt < 0 && line.indexOf('Running the Immrax robustness check') >= 0) {
            checkAt = i;
        }
        if (stage && candidateAt >= 0 && checkAt >= 0) break;
    }
    const started = job.started || 0;
    const elapsed = started ? Math.max(0, Date.now() / 1000 - started) : 0;
    const parts = ['Solving...', `${elapsed.toFixed(1)} s`];
    if (stage) parts.push(stage);
    if (candidate) parts.push(candidate);
    /* The check is the slow part, and the log says so once per candidate. */
    if (checkAt > candidateAt) parts.push('validating');
    return parts.join(' - ');
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
    /* The mat is the frame everything else is read against, so it is part of
       what "all" means - and a fresh studio then shows the field, not a void. */
    if (state.showMat) {
        points.push({ x: 0, y: 0 }, { x: state.mat.widthM, y: state.mat.heightM });
    }
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

/* The earlier solves, so a new trajectory can be compared with the ones that
   came before it. The history is the browser's session only: a reload
   starts a fresh field. */
const RUN_HISTORY_LIMIT = 8;

function rememberRun(samples) {
    if (!samples || samples.length < 2) return;
    state.runs.push({
        samples: samples.map((sample) => ({ x: sample.x, y: sample.y })),
        duration: samples[samples.length - 1].t,
    });
    while (state.runs.length > RUN_HISTORY_LIMIT) state.runs.shift();
    renderRunHistory();
}

function renderRunHistory() {
    const hint = el('run-history');
    if (!hint) return;
    if (!state.runs.length) {
        hint.textContent = 'No earlier runs yet.';
        return;
    }
    const last = state.runs[state.runs.length - 1].duration;
    hint.textContent = `${state.runs.length} earlier run`
        + `${state.runs.length === 1 ? '' : 's'} on the field`
        + ` (the last took ${last.toFixed(2)} s).`;
}

/* The canvas is drawn only when something changed. draw() asks for a
   frame; the drawing happens on the next animation frame. Without this the
   whole canvas - the mat, the models, the trajectory, the envelope and
   the eight earlier solves - was drawn sixty times a second, always. */
let canvasDirty = true;

function draw() {
    canvasDirty = true;
}

function drawNow() {
    canvasDirty = false;
    if (!ctx) return;
    const ratio = window.devicePixelRatio || 1;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawMat();
    drawFieldEdges();
    drawGrid();
    drawObstacles();

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

    drawRunHistory();
    /* The other candidates go under the ribbon, so the exported plan stays
       the thing the eye lands on. */
    drawCandidates();
    if (state.trajectory) {
        drawTrajectory(state.trajectory, ratio);
    }
    /* The envelope goes on top of the ribbon, not under it: the tube is a few
       millimetres wide (0.2 cm on the tick map), and under the ribbon's halo
       it disappeared altogether. */
    const envelopeOf = envelopeSamples();
    if (envelopeOf) drawEnvelope(envelopeOf);
    drawCollisionMarks();

    drawGhostPath();
    drawApexCues();
    drawWaypoints();
    drawMeasuredPoses();
    drawRobot();
    updateLegend();
    updatePlaybackLabel();
}

/* The mat under the mission. Its south-west corner is the field's (0, 0)
   and its north edge is up the screen, so the image is drawn from the two
   corners of the mat's rectangle in world coordinates. */
function drawMat() {
    const mat = state.mat;
    if (!state.showMat || !mat.image) return;
    const a = worldToScreen(0, 0);
    const b = worldToScreen(mat.widthM, mat.heightM);
    const x = Math.min(a.x, b.x);
    const y = Math.min(a.y, b.y);
    const width = Math.abs(b.x - a.x);
    const height = Math.abs(b.y - a.y);
    ctx.save();
    ctx.globalAlpha = 0.9;
    ctx.drawImage(mat.image, x, y, width, height);
    ctx.restore();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, width, height);
}

/* The field's walls: the mat's edge, and the launch areas' arcs. */
function drawFieldEdges() {
    const field = state.fields[state.season];
    if (!field) return;
    const ratio = window.devicePixelRatio || 1;
    const mat = field.mat || state.mat;

    ctx.save();
    ctx.strokeStyle = 'rgba(226, 232, 240, 0.35)';
    ctx.lineWidth = 2 * ratio;
    const corner = worldToScreen(0, 0);
    const far = worldToScreen(mat.width_m, mat.height_m);
    ctx.strokeRect(
        Math.min(corner.x, far.x), Math.min(corner.y, far.y),
        Math.abs(far.x - corner.x), Math.abs(far.y - corner.y));

    // The launch areas: quarter circles in the corners of the wall.
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.5)';
    (field.launch_areas || []).forEach((area) => {
        const start = area.quadrant === 2 ? Math.PI / 2 : 0;
        const end = area.quadrant === 2 ? Math.PI : Math.PI / 2;
        ctx.beginPath();
        for (let i = 0; i <= 24; i++) {
            const angle = start + (end - start) * i / 24;
            const point = worldToScreen(
                area.x_m + Math.cos(angle) * area.radius_m,
                area.y_m + Math.sin(angle) * area.radius_m);
            if (i === 0) ctx.moveTo(point.x, point.y);
            else ctx.lineTo(point.x, point.y);
        }
        ctx.stroke();
    });
    ctx.restore();
}

/* Which mission model is under the pointer?

   The test uses the same rotation the drawing does, so what lights up is
   what the pointer is on. */
function obstacleAt(x, y) {
    const field = state.fields[state.season];
    if (!field) return null;
    for (const obstacle of field.obstacles) {
        if (obstacle.kind === 'circle') {
            const dx = x - obstacle.x_m;
            const dy = y - obstacle.y_m;
            if (Math.hypot(dx, dy) <= obstacle.w_m / 2) return obstacle;
            continue;
        }
        const angle = obstacle.rot_deg * Math.PI / 180;
        const dx = x - obstacle.x_m;
        const dy = y - obstacle.y_m;
        const localX = dx * Math.cos(angle) + dy * Math.sin(angle);
        const localY = -dx * Math.sin(angle) + dy * Math.cos(angle);
        if (Math.abs(localX) <= obstacle.w_m / 2
                && Math.abs(localY) <= obstacle.h_m / 2) {
            return obstacle;
        }
    }
    return null;
}

/* Which waypoint is under the pointer? */
function waypointAt(x, y, tolerance) {
    for (let index = state.waypoints.length - 1; index >= 0; index--) {
        const point = state.waypoints[index];
        if (Math.hypot(point.x - x, point.y - y) <= tolerance) {
            return index;
        }
    }
    return -1;
}

/* The field: the mission models standing on the mat. They are measured off
   FIRST's public field wireframe, and the server owns the collision maths;
   the browser only draws what the server found. */
function drawObstacles() {
    const field = state.fields[state.season];
    if (!field) return;
    const ratio = window.devicePixelRatio || 1;
    const band = state.showPlacement ? (field.placement_uncertainty_m || 0) : 0;
    field.obstacles.forEach((obstacle) => {
        const half = { x: obstacle.w_m / 2, y: obstacle.h_m / 2 };
        const w = obstacle.w_m * camera.scale;
        const h = obstacle.h_m * camera.scale;
        const pad = band * camera.scale;
        const hovered = state.hoverObstacle === obstacle.id;
        ctx.save();
        ctx.translate(...Object.values(worldToScreen(obstacle.x_m, obstacle.y_m)));
        ctx.rotate(-obstacle.rot_deg * Math.PI / 180);
        /* The model's own position on the mat is not known to better than the
           field's placement uncertainty, so the dashed band around it is
           where a hit is only a maybe. */
        if (band > 0) {
            ctx.save();
            ctx.setLineDash([5 * ratio, 4 * ratio]);
            ctx.strokeStyle = 'rgba(251, 191, 36, 0.7)';
            ctx.lineWidth = 1.25 * ratio;
            if (obstacle.kind === 'circle') {
                ctx.beginPath();
                ctx.arc(0, 0, half.x * camera.scale + pad, 0, Math.PI * 2);
                ctx.stroke();
            } else {
                ctx.strokeRect(-w / 2 - pad, -h / 2 - pad, w + 2 * pad, h + 2 * pad);
            }
            ctx.restore();
        }
        if (hovered) {
            ctx.strokeStyle = 'rgba(56, 189, 248, 0.95)';
            ctx.fillStyle = 'rgba(56, 189, 248, 0.25)';
            ctx.lineWidth = 2.5 * ratio;
        } else {
            ctx.strokeStyle = 'rgba(148, 163, 184, 0.55)';
            ctx.fillStyle = 'rgba(148, 163, 184, 0.14)';
            ctx.lineWidth = 1.5 * ratio;
        }
        if (obstacle.kind === 'circle') {
            ctx.beginPath();
            ctx.arc(0, 0, half.x * camera.scale, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
        } else {
            ctx.fillRect(-w / 2, -h / 2, w, h);
            ctx.strokeRect(-w / 2, -h / 2, w, h);
        }
        ctx.restore();
        if (hovered) {
            const at = worldToScreen(obstacle.x_m, obstacle.y_m);
            ctx.save();
            ctx.font = `${12 * ratio}px system-ui, sans-serif`;
            ctx.fillStyle = '#38bdf8';
            ctx.textAlign = 'center';
            const name = band > 0
                ? `${obstacle.name} ±${Math.round(band * 1000)} mm` : obstacle.name;
            ctx.fillText(name, at.x, at.y - 10 * ratio);
            ctx.restore();
        }
    });
}

/* Where the planned path ran into something. The mark is the sample's
   position; the badge names what it hit. */
function drawCollisionMarks() {
    const ratio = window.devicePixelRatio || 1;
    state.collisions.forEach((collision) => {
        const point = worldToScreen(collision.x_m, collision.y_m);
        const shallow = collision.uncertain;
        ctx.save();
        ctx.strokeStyle = shallow ? 'rgba(251, 191, 36, 0.95)' : 'rgba(248, 113, 113, 0.95)';
        ctx.fillStyle = shallow ? 'rgba(251, 191, 36, 0.25)' : 'rgba(248, 113, 113, 0.25)';
        ctx.lineWidth = 2.5 * ratio;
        ctx.beginPath();
        ctx.arc(point.x, point.y, 11 * ratio, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(point.x - 6 * ratio, point.y - 6 * ratio);
        ctx.lineTo(point.x + 6 * ratio, point.y + 6 * ratio);
        ctx.moveTo(point.x + 6 * ratio, point.y - 6 * ratio);
        ctx.lineTo(point.x - 6 * ratio, point.y + 6 * ratio);
        ctx.stroke();
        ctx.font = `${12 * ratio}px system-ui, sans-serif`;
        ctx.fillStyle = ctx.strokeStyle;
        ctx.textAlign = 'left';
        ctx.fillText(collision.name, point.x + 16 * ratio, point.y - 10 * ratio);
        ctx.restore();
    });
}

/* The earlier solves, as faint dashed paths, so a new trajectory can be
   compared with the ones that came before it. */
function drawRunHistory() {
    if (!state.showRuns) return;
    const ratio = window.devicePixelRatio || 1;
    ctx.save();
    ctx.setLineDash([4 * ratio, 5 * ratio]);
    ctx.lineWidth = 1.5 * ratio;
    state.runs.forEach((run, index) => {
        const fade = 0.55 - 0.05 * index;
        ctx.strokeStyle = `rgba(148, 163, 184, ${Math.max(fade, 0.15)})`;
        const samples = run.samples || [];
        if (samples.length < 2) return;
        ctx.beginPath();
        samples.forEach((sample, i) => {
            const p = worldToScreen(sample.x, sample.y);
            if (i === 0) ctx.moveTo(p.x, p.y);
            else ctx.lineTo(p.x, p.y);
        });
        ctx.stroke();
    });
    ctx.restore();
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

/* The colour of one candidate, by its seed. The triad keeps three hues; a
   rescue seed is amber, the same as the rescue chip. */
const CANDIDATE_COLORS = {
    frontier_sprint: '#38bdf8',
    frontier_sweeper: '#a78bfa',
    frontier_exploratory: '#34d399',
};

function candidateColor(style) {
    return CANDIDATE_COLORS[style] || '#fbbf24';
}

/* Every solved candidate's path, so the comparison is on the field and not
   only in a table. The plan that was exported keeps its own ribbon; the
   candidate the chips point at is drawn stronger than the rest. */
function drawCandidates() {
    const entries = state.candidatePaths || [];
    if (entries.length < 2) return;
    const ratio = window.devicePixelRatio || 1;
    const report = state.validationReport;
    const exported = report ? (report.selected || 1) - 1 : -1;
    const active = state.reportCandidate !== null
        ? state.reportCandidate
        : exported;
    ctx.save();
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    entries.forEach((entry, index) => {
        const samples = entry.samples || [];
        if (samples.length < 2) return;
        /* The exported plan is the bright ribbon; drawing it twice would hide
           the saturation colouring under a plain line. */
        if (index === exported && state.trajectory) return;
        ctx.globalAlpha = index === active ? 0.95 : 0.4;
        drawPath(samples, candidateColor(entry.style),
            (index === active ? 2.5 : 1.5) * ratio);
    });
    ctx.restore();
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

/* The short way round, as the firmware takes a heading difference. */
function wrapAngle(angle) {
    let value = Number(angle) || 0;
    while (value > Math.PI) value -= 2 * Math.PI;
    while (value < -Math.PI) value += 2 * Math.PI;
    return value;
}

/* How close a sample is to the actuator diamond: it asks for
   |v| + (W/2)|omega| of the wheel speed the motors can give, which is the
   same as the faster wheel's share of the top speed. Below 0.85 the plan is
   clear of the limit, 0.85 to 1 rides it, and above 1 asks for more than the
   wheels have. */
function actuatorLimit() {
    const limits = state.jobResult && state.jobResult.limits;
    const speed = limits ? Number(limits.speed_limit_m_s) : NaN;
    if (Number.isFinite(speed) && speed > 0) return speed;
    return topSpeed();
}

function trackWidth() {
    return Number(robotFields().track_width) || 0.0965;
}

function actuatorSaturation(sample, limit, track = trackWidth()) {
    if (!limit || !sample) return 0;
    const vl = Number(sample.vl) || 0;
    const vr = Number(sample.vr) || 0;
    const omega = Number.isFinite(Number(sample.omega))
        ? Number(sample.omega) : (vr - vl) / track;
    return (Math.abs((vl + vr) / 2) + (track / 2) * Math.abs(omega)) / limit;
}

/* The trajectory, coloured by how hard it asks the motors to work. The halo
   is the same path wider, so the thin ribbon stays visible over the mat. */
function drawTrajectory(samples, ratio) {
    drawPath(samples, 'rgba(56, 189, 248, 0.25)', 10 * ratio);
    const limit = state.showSaturation ? actuatorLimit() : null;
    if (!limit) {
        drawPath(samples, '#38bdf8', 2.5 * ratio);
        return;
    }
    const track = trackWidth();
    ctx.save();
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.lineWidth = 2.5 * ratio;
    for (let i = 0; i < samples.length - 1; i++) {
        const saturation = Math.max(actuatorSaturation(samples[i], limit, track),
            actuatorSaturation(samples[i + 1], limit, track));
        ctx.strokeStyle = saturation > 1.0 ? 'rgba(248, 113, 113, 0.95)'
            : (saturation >= 0.85 ? 'rgba(251, 191, 36, 0.95)' : '#38bdf8');
        const a = worldToScreen(samples[i].x, samples[i].y);
        const b = worldToScreen(samples[i + 1].x, samples[i + 1].y);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
    }
    ctx.restore();
}

/* The late-apex cue, and the line a free-heading waypoint lets the robot
   keep. A turning segment is clipped at about 68% of its length - the apex
   a driver aims at when the corner opens on exit - and a waypoint whose
   heading is free is crossed along the bisector of the two tangents, which
   is the direction that keeps the momentum. */
function drawApexCues() {
    if (!state.showApex || !state.trajectory || state.trajectory.length < 2) return;
    const ratio = window.devicePixelRatio || 1;
    const samples = state.trajectory;
    const bounds = segmentBounds();

    ctx.save();
    curvedSegments().forEach((segment) => {
        const sample = sampleAtFraction(segment.from, segment.to, 0.68);
        const p = worldToScreen(sample.x, sample.y);
        const size = 6 * ratio;
        ctx.fillStyle = 'rgba(245, 158, 11, 0.9)';
        ctx.strokeStyle = 'rgba(245, 158, 11, 0.9)';
        ctx.lineWidth = 1.2 * ratio;
        ctx.beginPath();
        ctx.moveTo(p.x, p.y - size);
        ctx.lineTo(p.x + size, p.y);
        ctx.lineTo(p.x, p.y + size);
        ctx.lineTo(p.x - size, p.y);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
    });

    ctx.strokeStyle = 'rgba(226, 232, 240, 0.55)';
    ctx.lineWidth = 1.5 * ratio;
    bisectorWaypoints().forEach((point) => {
        const index = state.waypoints.indexOf(point);
        const before = bounds[index];
        const after = bounds[index + 1];
        if (before === undefined || after === undefined
                || before >= samples.length || after >= samples.length) return;
        const previous = samples[Math.max(before - 1, 0)];
        const next = samples[Math.min(after + 1, samples.length - 1)];
        const incoming = Math.atan2(samples[before].y - previous.y,
            samples[before].x - previous.x);
        const outgoing = Math.atan2(next.y - samples[after].y,
            next.x - samples[after].x);
        const angle = (incoming + wrapAngle(outgoing - incoming) / 2);
        const start = worldToScreen(point.x, point.y);
        const length = 28 * ratio;
        const end = {
            x: start.x + Math.cos(angle) * length,
            y: start.y - Math.sin(angle) * length,
        };
        ctx.beginPath();
        ctx.moveTo(start.x, start.y);
        ctx.lineTo(end.x, end.y);
        ctx.stroke();
        [-0.45, 0.45].forEach((spread) => {
            ctx.beginPath();
            ctx.moveTo(end.x, end.y);
            ctx.lineTo(end.x - Math.cos(angle + spread) * 6 * ratio,
                end.y + Math.sin(angle + spread) * 6 * ratio);
            ctx.stroke();
        });
    });
    ctx.restore();
}

/* The samples whose reachable set is on the field: the exported plan when
   there is one, otherwise the candidate the chips point at. A failed
   validation still wrote every attempt's tube, so the key keeps its meaning
   there too. */
function envelopeSamples() {
    if (state.trajectory && state.trajectory.length) return state.trajectory;
    const entries = state.candidatePaths || [];
    if (!entries.length) return null;
    const report = state.validationReport;
    const index = state.reportCandidate !== null
        ? state.reportCandidate
        : (report ? (report.selected || 1) - 1 : 0);
    const entry = entries[index] || entries[0];
    return entry && entry.samples && entry.samples.length ? entry.samples : null;
}

function drawEnvelope(samples) {
    if (!state.showEnvelope) return;
    if (state.showEnvelopeBoxes) {
        drawEnvelopeBoxes(samples);
        return;
    }
    const withEnv = samples.filter((s) => s.reachability_envelope);
    if (withEnv.length < 2) return;
    const ratio = window.devicePixelRatio || 1;
    ctx.save();
    ctx.fillStyle = 'rgba(52, 211, 153, 0.10)';
    ctx.strokeStyle = 'rgba(52, 211, 153, 0.75)';
    ctx.lineWidth = 1.5 * ratio;
    ctx.setLineDash([3 * ratio, 3 * ratio]);
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
    ctx.restore();
}

/* The same tube drawn as what the validator stores: one axis-aligned box per
   sample. The boxes go into one path and are filled once, so overlapping
   boxes read as a single band and no outline can fold back on itself. The
   corner ribbon above does fold: it joins two opposite corners of each box,
   so wherever the path turns the join runs to the other side of the band and
   the fill crosses over. That crossing is the drawing, not the robot. */
function drawEnvelopeBoxes(samples) {
    const withEnv = samples.filter((s) => s.reachability_envelope);
    if (withEnv.length < 2) return;
    const ratio = window.devicePixelRatio || 1;
    ctx.save();
    ctx.fillStyle = 'rgba(52, 211, 153, 0.14)';
    ctx.strokeStyle = 'rgba(52, 211, 153, 0.45)';
    ctx.lineWidth = 0.75 * ratio;
    ctx.beginPath();
    withEnv.forEach((sample) => {
        const env = sample.reachability_envelope;
        const a = worldToScreen(env.x_min, env.y_min);
        const b = worldToScreen(env.x_max, env.y_max);
        ctx.rect(a.x, a.y, b.x - a.x, b.y - a.y);
    });
    ctx.fill();
    /* Struck separately: a stroke on the shared path would outline every
       seam between two boxes and read as a grid. */
    withEnv.forEach((sample) => {
        const env = sample.reachability_envelope;
        const a = worldToScreen(env.x_min, env.y_min);
        const b = worldToScreen(env.x_max, env.y_max);
        ctx.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y);
    });
    ctx.restore();
}

function drawWaypoints() {
    const ratio = window.devicePixelRatio || 1;
    state.waypoints.forEach((point, index) => {
        const p = worldToScreen(point.x, point.y);
        const active = index === state.selected;
        const hovered = index === state.hoverWaypoint;
        const failing = state.failingSegments.has(index) || state.failingSegments.has(index - 1);
        ctx.fillStyle = active ? '#38bdf8'
            : (hovered ? '#7dd3fc'
                : (point.heading === null ? '#94a3b8' : '#f472b6'));
        ctx.beginPath();
        ctx.arc(p.x, p.y, (active ? 7 : 5.5) * ratio, 0, Math.PI * 2);
        ctx.fill();
        if (hovered && !active) {
            ctx.strokeStyle = 'rgba(56, 189, 248, 0.8)';
            ctx.lineWidth = 2 * ratio;
            ctx.beginPath();
            ctx.arc(p.x, p.y, 9 * ratio, 0, Math.PI * 2);
            ctx.stroke();
        }

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

        drawWaypointBadges(point, p, ratio);

        ctx.fillStyle = '#94a3b8';
        ctx.font = `${11 * ratio}px system-ui, sans-serif`;
        ctx.fillText(`W${index}`, p.x + 10 * ratio, p.y - 8 * ratio);
    });
}

/* A stop is a square, an event a diamond, both with their name next to them. */
function drawWaypointBadges(point, p, ratio) {
    if (!point.stop && !point.event) return;
    const above = p.y - 24 * ratio;
    ctx.save();
    if (point.stop) {
        ctx.fillStyle = '#38bdf8';
        const size = 5 * ratio;
        ctx.fillRect(p.x - size / 2, above - size / 2, size, size);
    }
    if (point.event) {
        ctx.fillStyle = '#f472b6';
        const size = 6 * ratio;
        ctx.beginPath();
        ctx.moveTo(p.x, above - size);
        ctx.lineTo(p.x + size, above);
        ctx.lineTo(p.x, above + size);
        ctx.lineTo(p.x - size, above);
        ctx.closePath();
        ctx.fill();
    }
    const labels = [];
    if (point.stop) labels.push('stop');
    if (point.event) labels.push(point.event);
    ctx.font = `bold ${10 * ratio}px system-ui, sans-serif`;
    ctx.fillStyle = point.event ? '#f9a8d4' : '#7dd3fc';
    ctx.textAlign = 'left';
    ctx.fillText(labels.join(' '), p.x + 11 * ratio, above + 4 * ratio);
    ctx.textAlign = 'start';
    ctx.restore();
}

/* Where the robot actually ended, next to where the plan said it would. */
function drawMeasuredPoses() {
    if (!state.measuredPoses.length) return;
    const ratio = window.devicePixelRatio || 1;
    const last = state.trajectory ? state.trajectory[state.trajectory.length - 1] : null;
    state.measuredPoses.forEach((pose) => {
        const p = measuredInPlanFrame(pose);
        if (!p) return;
        const s = worldToScreen(p.x, p.y);
        if (last) {
            const planned = worldToScreen(last.x, last.y);
            ctx.save();
            ctx.setLineDash([4 * ratio, 4 * ratio]);
            ctx.strokeStyle = 'rgba(52, 211, 153, 0.7)';
            ctx.lineWidth = 1.5 * ratio;
            ctx.beginPath();
            ctx.moveTo(planned.x, planned.y);
            ctx.lineTo(s.x, s.y);
            ctx.stroke();
            ctx.restore();
        }
        ctx.save();
        ctx.strokeStyle = '#34d399';
        ctx.lineWidth = 2 * ratio;
        ctx.beginPath();
        ctx.moveTo(s.x - 9 * ratio, s.y);
        ctx.lineTo(s.x + 9 * ratio, s.y);
        ctx.moveTo(s.x, s.y - 9 * ratio);
        ctx.lineTo(s.x, s.y + 9 * ratio);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(s.x, s.y, 6 * ratio, 0, Math.PI * 2);
        ctx.stroke();
        const angle = p.heading_deg * Math.PI / 180;
        ctx.beginPath();
        ctx.moveTo(s.x, s.y);
        ctx.lineTo(s.x + Math.cos(angle) * 20 * ratio, s.y - Math.sin(angle) * 20 * ratio);
        ctx.stroke();
        ctx.restore();
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
    if (!drag) {
        // Nothing is being dragged, so this is a hover: light up whatever
        // is under the pointer.
        const world = screenToWorld(event.offsetX * ratio, event.offsetY * ratio);
        const obstacle = obstacleAt(world.x, world.y);
        const waypoint = waypointAt(world.x, world.y, 8 / camera.scale);
        const obstacleId = obstacle ? obstacle.id : null;
        const waypointIndex = waypoint;
        if (obstacleId !== state.hoverObstacle
                || waypointIndex !== state.hoverWaypoint) {
            state.hoverObstacle = obstacleId;
            state.hoverWaypoint = waypointIndex;
            // A waypoint drags, a model answers a hover, and the empty
            // field takes a double-click to place one.
            canvas.style.cursor = waypointIndex >= 0 ? 'move'
                : obstacleId !== null ? 'pointer' : 'crosshair';
            draw();
        }
        return;
    }
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
            markMissionChanged();
            renderEditor();
        }
    }
    draw();
});

canvas.addEventListener('pointerleave', () => {
    if (state.hoverObstacle === null && state.hoverWaypoint < 0) return;
    state.hoverObstacle = null;
    state.hoverWaypoint = -1;
    canvas.style.cursor = '';
    draw();
});

canvas.addEventListener('pointerup', () => {
    if (drag && drag.kind === 'waypoint' && drag.moved) {
        // One history entry per drag, not one per mouse move.
        updatePathSummary();
        pushHistory();
    } else if (drag && drag.kind === 'waypoint' && !drag.moved) {
        // A click, not a drag: the user picked this waypoint to work with
        // it, so the editor comes forward.
        showPanel('panel-waypoint');
    }
    drag = null;
    draw();
});

/* A double click on the mat adds a waypoint. A single click used to, and a
   stray click on the mat placed one nobody meant. */
canvas.addEventListener('dblclick', (event) => {
    const ratio = window.devicePixelRatio || 1;
    if (hitTest(event.offsetX * ratio, event.offsetY * ratio) >= 0) return;
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
    markMissionChanged();
    renderAll();
    pushHistory();
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
    // Shift is a coarse step, Alt a fine one.
    const step = event.shiftKey ? NUDGE_STEP * 10 : (event.altKey ? NUDGE_STEP / 10 : NUDGE_STEP);
    if (event.key === ' ') {
        event.preventDefault();
        el('btn-play').click();
        return;
    }
    const nudges = {
        ArrowLeft: [-step, 0],
        ArrowRight: [step, 0],
        ArrowUp: [0, step],
        ArrowDown: [0, -step],
    };
    if (nudges[event.key] && state.selected >= 0) {
        event.preventDefault();
        const [dx, dy] = nudges[event.key];
        nudgeSelected(dx, dy);
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
    markMissionChanged();
    renderAll();
    pushHistory();
}

function updateField(field, value) {
    const point = currentWaypoint();
    if (!point) return;
    if (field === 'x' || field === 'y') {
        // An empty field is a half-typed number, not a zero: deleting the
        // contents of X must not move the waypoint to the origin. A stray
        // character is not a number either, and the mission file would
        // write it as null and the server would reject the solve.
        const raw = String(value === null || value === undefined ? '' : value).trim();
        if (raw === '') return;
        const number = Number(raw);
        if (!Number.isFinite(number)) return;
        value = number;
    } else if (typeof value === 'number' && !Number.isFinite(value)) {
        return;
    }
    point[field] = value;
    markMissionChanged();
    renderWaypointList();
    updateMissionState();
}

el('wp-x').addEventListener('input', (e) => updateField('x', e.target.value));
el('wp-y').addEventListener('input', (e) => updateField('y', e.target.value));
el('wp-heading').addEventListener('input', (e) => {
    const point = currentWaypoint();
    if (!point) return;
    const degrees = Number(e.target.value);
    point.heading = Number.isFinite(degrees) ? degrees * Math.PI / 180 : null;
    markMissionChanged();
    renderWaypointList();
    updateMissionState();
    draw();
});
el('wp-free-heading').addEventListener('change', (e) => {
    const point = currentWaypoint();
    if (!point) return;
    point.heading = e.target.checked ? null : 0;
    markMissionChanged();
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
    markMissionChanged();
    renderAll();
    pushHistory();
}

el('btn-solve').addEventListener('click', solve);
el('btn-validate').addEventListener('click', () => {
    /* The lever the result names: turn the validator on and solve again.
       The Solve tab comes forward, so the switch that changed is visible. */
    el('opt-validate').checked = true;
    updateGateFields();
    showPanel('panel-solve');
    solve();
});
el('btn-report').addEventListener('click', openMissionReport);
el('btn-download-project').addEventListener('click', downloadProject);
el('btn-upload-project').addEventListener('click', () => el('file-project').click());
el('file-project').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) openProjectFile(file);
    e.target.value = '';
});
el('btn-push').addEventListener('click', pushToHub);
el('btn-scan').addEventListener('click', scanHubs);
renderPlotTabs();
el('btn-collect').addEventListener('click', startCollect);
el('btn-save-sheet').addEventListener('click', saveSheet);
el('btn-fit').addEventListener('click', fitModel);
el('cal-name').addEventListener('change', () => { loadSheet(); });
el('btn-load-model').addEventListener('click', () => el('file-model').click());
el('file-model').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) await loadModelFromFile(file);
    e.target.value = '';
});
el('btn-load-model-path').addEventListener('click', loadModelFromPath);
el('btn-use-model').addEventListener('click', () => {
    const path = el('model-list').value;
    const model = (state.models || []).find((m) => m.path === path);
    if (model) useModel(model);
});
el('opt-use-model').addEventListener('change', (e) => {
    state.useModel = e.target.checked && Boolean(state.model);
    if (state.useModel) {
        el('opt-validate').checked = true;
        updateGateFields();
    }
    scheduleSessionSave();
});
el('btn-use-robot-geometry').addEventListener('click', useRobotGeometry);
el('hub-conn').addEventListener('change', (e) => {
    state.hubConn = e.target.value;
    scheduleSessionSave();
});
el('hub-name').addEventListener('input', (e) => {
    state.hubName = e.target.value.trim();
    scheduleSessionSave();
});
el('btn-cancel').addEventListener('click', async () => {
    if (!state.jobId) return;
    try {
        await api(`/api/jobs/${state.jobId}/cancel`, { method: 'POST' });
    } catch (err) {
        toast(err.message, true);
    }
});

el('btn-frame').addEventListener('click', () => { frameAll(); draw(); });
el('field-season').addEventListener('change', (e) => {
    state.season = e.target.value;
    state.collisions = [];
    /* The mat belongs to the season: the old picture would sit under the
       new field's models. The new one comes by itself. */
    state.mat.image = null;
    el('mat-name').textContent = 'No mat image.';
    scheduleSessionSave();
    draw();
    fetchMat(true);
});

/* The mat image is the user's: FIRST's artwork is not ours to ship. The
   image is not kept across a reload (a browser cannot store a file the
   user picked without asking), but the size is. */
el('file-mat').addEventListener('change', (event) => {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
        state.mat.image = image;
        state.mat.name = file.name;
        el('mat-name').textContent = `Mat: ${file.name}`;
        draw();
    };
    image.onerror = () => toast(`Could not read ${file.name}.`, true);
    image.src = url;
});

/* Fetch the season's mat picture. FIRST's mat artwork is not ours to ship,
   so the server downloads it on request and keeps it outside the repo. It
   runs by itself on load and on a season change: a blank mat is the first
   thing an operator sees, and the server caches the picture in mats/ after
   the first download. */
async function fetchMat(quiet) {
    if (!state.season) {
        if (!quiet) toast('Pick a season first.', true);
        return false;
    }
    const button = el('btn-fetch-mat');
    button.disabled = true;
    button.textContent = 'Downloading...';
    try {
        const result = await api('/api/field/mat', {
            method: 'POST',
            body: JSON.stringify({ season: state.season }),
        });
        // The mat's size comes from the field, so the picture and the
        // mission models line up.
        state.mat.widthM = result.width_m;
        state.mat.heightM = result.height_m;
        el('mat-width').value = String(Math.round(result.width_m * 1000));
        el('mat-height').value = String(Math.round(result.height_m * 1000));
        el('mat-credit').textContent = result.credit || '';
        await loadMatImage(result.url);
        if (!quiet) toast('The mat is ready.');
        return true;
    } catch (err) {
        // A failed automatic download still has to be said out loud: the
        // field then draws without a mat, and that is worth knowing.
        toast(err.message, true);
        return false;
    } finally {
        button.disabled = false;
        button.textContent = 'Download the mat';
    }
}
el('btn-fetch-mat').addEventListener('click', () => fetchMat(false));

function loadMatImage(url) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => {
            state.mat.image = image;
            state.mat.name = url.split('/').pop() || url;
            el('mat-name').textContent = `Mat: ${state.mat.name}`;
            draw();
            resolve(image);
        };
        image.onerror = () => reject(new Error('Could not read the mat.'));
        image.src = url;
    });
}
el('mat-width').addEventListener('change', (e) => {
    const value = Number(e.target.value);
    if (Number.isFinite(value) && value > 0) state.mat.widthM = value / 1000;
    scheduleSessionSave();
    draw();
});
el('mat-height').addEventListener('change', (e) => {
    const value = Number(e.target.value);
    if (Number.isFinite(value) && value > 0) state.mat.heightM = value / 1000;
    scheduleSessionSave();
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
el('btn-copy-log').addEventListener('click', copyLog);

/* The log is what an operator pastes into an issue, so the copy takes the
   whole thing rather than the selection. The clipboard needs a secure
   context, so a plain http page selects the text instead and Ctrl+C finishes
   it. */
async function copyLog() {
    const node = el('log');
    const text = node.textContent || '';
    if (!text.trim()) {
        toast('The log is empty.');
        return;
    }
    try {
        await navigator.clipboard.writeText(text);
        toast('The solver log is on the clipboard.');
    } catch (err) {
        const range = document.createRange();
        range.selectNodeContents(node);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        toast('Clipboard blocked here: the log is selected, press Ctrl+C.');
    }
}
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
                mission: missionDocument(),
            }),
        });
        /* What is on disk now, so the studio can say whether the mission on
           the screen is still the one that was saved. */
        state.savedFingerprint = missionFingerprint(false);
        state.savedName = result.name;
        updateMissionState();
        renderHistory(result.history || []);
        const committed = result.commit
            ? ` (commit ${result.commit.slice(0, 7)})`
            : '';
        /* A save that did not become a commit is not in the history. Say so:
           the file is on disk, but the version the operator can go back to
           is not. */
        if (result.commit_error) {
            toast(`Saved missions/${result.name}.json - NOT in the history: `
                + 'git failed', true);
        } else {
            toast(`Saved missions/${result.name}.json${committed}`);
        }
        await loadExamples();
    } catch (err) {
        toast(err.message, true);
    }
}

/* ------------------------------------------------- the mission's history */

/* The missions folder is a git repository, so every save is a commit and an
 * earlier version can be brought back. */
function renderHistory(entries) {
    const list = el('history-list');
    list.innerHTML = '';
    if (!entries.length) {
        list.innerHTML = '<p class="hint">No history yet. Saving the mission makes the '
            + 'first commit.</p>';
        return;
    }
    entries.forEach((entry) => {
        const row = document.createElement('div');
        row.className = 'history-row';
        const text = document.createElement('span');
        const when = (entry.when || '').slice(0, 16).replace('T', ' ');
        text.textContent = `${when} - ${entry.message}`;
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'btn small';
        button.textContent = 'Restore';
        button.addEventListener('click', () => restoreMission(entry.sha));
        row.appendChild(text);
        row.appendChild(button);
        list.appendChild(row);
    });
}

async function loadHistory(name) {
    try {
        const payload = await api(
            `/api/missions/history?name=${encodeURIComponent(name || '')}`);
        renderHistory(payload.history || []);
    } catch (err) {
        const list = el('history-list');
        list.textContent = '';
        const hint = document.createElement('p');
        hint.className = 'hint';
        hint.textContent = err.message;
        list.appendChild(hint);
    }
}

async function restoreMission(sha) {
    try {
        const payload = await api('/api/missions/restore', {
            method: 'POST',
            body: JSON.stringify({ name: el('mission-name').value, sha }),
        });
        state.missionScoring = payload.scoring || {};
        setWaypoints(payload.waypoints);
        renderHistory(payload.history || []);
        toast('Restored that version.');
    } catch (err) {
        toast(err.message, true);
    }
}

function loadMissionDocument(data) {
    if (Array.isArray(data)) {
        state.missionScoring = {};
        return data;
    }
    if (data && Array.isArray(data.waypoints)) {
        state.missionScoring = data.scoring && typeof data.scoring === 'object'
            ? data.scoring : {};
        return data.waypoints;
    }
    throw new Error('A mission file must contain waypoints.');
}

el('mission-list').addEventListener('change', (e) => {
    const mission = (state.missions || []).find((m) => m.name === e.target.value);
    if (!mission) return;
    loadHistory(mission.name);
    // The previous mission stays in the undo history, so nothing is lost.
    setWaypoints(loadMissionDocument(mission.data));
    el('mission-name').value = mission.name;
    frameAll();
    draw();
    scheduleSessionSave();
});

el('config-list').addEventListener('change', (e) => {
    const config = (state.configs || []).find((c) => c.name === e.target.value);
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
        const waypoints = loadMissionDocument(data);
        setWaypoints(waypoints);
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
    updateGateFields();
    scheduleSessionSave();
});

['opt-samples', 'opt-accuracy', 'opt-candidates', 'opt-export', 'opt-controller-dt',
 'opt-force-weight', 'opt-propagation', 'opt-fast-polish', 'opt-best-effort',
 'opt-envelope-limit', 'opt-envelope-limit-per-metre',
 'opt-score-profile', 'opt-target-time', 'opt-max-time', 'opt-jerk-target',
 'opt-jerk-max', 'opt-accept-score']
    .forEach((id) => el(id).addEventListener('change', () => {
        updateGateFields();
        syncPresetRadio();
        scheduleSessionSave();
    }));

el('opt-auto-mesh').addEventListener('change', () => {
    updateMeshField();
    scheduleSessionSave();
});

el('mission-name').addEventListener('input', scheduleSessionSave);

window.addEventListener('resize', () => { resize(); drawPlot(); });

/* ------------------------------------------------------------------ loop */

function animate() {
    if (playback.playing && playback.duration > 0) {
        playback.time += 0.02;
        if (playback.time > playback.duration) playback.time = 0;
        el('playback').value = Math.round((playback.time / playback.duration) * 1000);
        // The robot marker moves, so the canvas has to be redrawn.
        canvasDirty = true;
    }
    if (canvasDirty) drawNow();
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
        setupTabs();
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
        // --live passes a mission and options on the command line: those win
        // over whatever this browser had open.
        const fromCli = Boolean(initial.auto_solve);
        if (fromCli && initial.waypoints) {
            state.missionScoring = initial.scoring || {};
            setWaypoints(initial.waypoints);
            el('mission-name').value = state.configName || 'mission';
            if (initial.config) {
                state.config = initial.config;
                state.configName = initial.config_name || '';
                state.configEdited = false;
            }
            applyOptionsToForm(initial.options);
            if (initial.options && initial.options.hub_conn) {
                state.hubConn = initial.options.hub_conn;
            }
            if (initial.options && initial.options.hub_name) {
                state.hubName = initial.options.hub_name;
            }
        } else if (restored) {
            setWaypoints(session.waypoints);
            el('mission-name').value = session.name || 'mission';
            if (session.config) {
                state.config = session.config;
                state.configName = session.configName || '';
                state.configEdited = Boolean(session.configEdited);
            }
            applyOptionsToForm(session.options);
            if (session.hubConn) state.hubConn = session.hubConn;
            if (session.hubName) state.hubName = session.hubName;
            if (session.model) {
                state.model = session.model;
                state.useModel = Boolean(session.useModel);
            }
            if (Number.isFinite(session.matWidthM) && session.matWidthM > 0) {
                state.mat.widthM = session.matWidthM;
            }
            if (Number.isFinite(session.matHeightM) && session.matHeightM > 0) {
                state.mat.heightM = session.matHeightM;
            }
            if (session.season) {
                state.season = session.season;
            }
            if (session.overlays) {
                OVERLAY_FLAGS.forEach((key) => {
                    if (typeof session.overlays[key] === 'boolean') {
                        state[key] = session.overlays[key];
                    }
                });
            }
        } else if (initial.waypoints) {
            state.missionScoring = initial.scoring || {};
            setWaypoints(initial.waypoints);
            el('mission-name').value = state.configName || 'mission';
            if (initial.options && Object.keys(initial.options).length) {
                applyOptionsToForm(initial.options);
            } else {
                applyPreset('fast');
            }
        } else {
            setWaypoints([blankWaypoint(0, 0, 0)]);
            applyPreset('fast');
        }
        resetHistory();

        await loadExamples();
        loadHistory(state.savedName || el('mission-name').value);
        renderRobotFacts();
        renderRobotEditor();
        renderRobotChecks();
        el('hub-conn').value = state.hubConn;
        el('hub-name').value = state.hubName;
        el('opt-use-model').checked = state.useModel && Boolean(state.model);
        renderPlotTabs();
        renderLegend();
        drawPlot();
        if (!el('cal-name').dataset.touched) {
            el('cal-name').value = state.configName || 'robot';
        }
        renderModel();
        updatePushButton();
        updateGateFields();
        updateMeshField();
        el('mat-width').value = String(Math.round(state.mat.widthM * 1000));
        el('mat-height').value = String(Math.round(state.mat.heightM * 1000));
        renderRunHistory();
        try {
            state.robotInfo = await api('/api/robot');
            renderRobotChecks();
        } catch (err) {
            // The robot's files are optional; the studio works without them.
        }
        try {
            const models = await api('/api/robot/models');
            state.models = models.models || [];
            await loadModels();
            if (state.model && !state.models.some((model) => model.path === state.model.path)) {
                // The file is gone; do not keep pointing at it.
                state.model = null;
                state.useModel = false;
                el('opt-use-model').checked = false;
                renderModel();
                toast('The measured model file is gone; the nominal model is used.');
            }
        } catch (err) {
            // No measured model yet; the nominal one is used.
        }
        // A robot with a collected session reopens with its sheet.
        loadSheet();
        // The field: the mission models standing on the mat.
        await loadFields();
        if (session && session.season) {
            state.season = session.season;
            const select = el('field-season');
            if (select) select.value = state.season;
        }
        /* The mat picture comes by itself, so a new studio is a field and not
           a void. It is not awaited: the models draw first, the picture lands
           when the server has it (and it is cached there after that). */
        fetchMat(true);

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
        /* A solve that is already running - another tab, or a reload in the
           middle of one - is attached to, not started again. The server
           refuses a second solve, so a studio that did not ask would look
           stuck: every Solve press would come back with "a solve is
           already running" and the Cancel button would have nothing to
           cancel. */
        let running = null;
        try {
            const current = await api('/api/jobs/current');
            const job = current && current.job;
            /* Only a solve is adopted: a push or a calibration run is not
               this studio's solve, and polling it as one would look for a
               trajectory it does not have. */
            if (job && job.status === 'running' && (!job.kind || job.kind === 'solve')) {
                running = job;
            }
        } catch (err) {
            // The server cannot say: the studio starts its own solve.
        }
        if (running) {
            state.jobId = running.id;
            state.reportCandidate = null;
            /* The mission on the screen is the one the job is solving, as
               far as the studio can tell: the job carries the name of the
               mission it was given, and the server slugs that name. A job
               for another mission keeps no fingerprint, so its trajectory
               is not offered as the answer to this one. */
            const sameMission = sameMissionName(running.name, el('mission-name').value);
            state.solvingFingerprint = sameMission ? missionFingerprint() : null;
            setBusy(true);
            el('result-status').textContent = 'Solving...';
            setStatus('Attached to the solve that is already running');
            pollJob();
        } else if (initial.auto_solve) {
            // --live: the studio opens on this mission and solves it right away.
            solve();
        }
    } catch (err) {
        setStatus(err.message, 'error');
        toast(err.message, true);
    }
}

init();
