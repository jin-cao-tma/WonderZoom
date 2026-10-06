// WonderZoom streaming viewer.
//
// Shared by two pages:
//   - index_stream.html: render-only viewer (run_render_only.py), usually opened
//     as a local file:// page with an SSH tunnel to localhost:7747.
//   - index_gen.html: generation UI, served by run.py at '/'.
// Every DOM element below is optional; all accesses must stay null-safe.
//
// Server URL: ?server=<host:port | url>, else window.WZ_DEFAULT_SERVER (set by
// index_gen.html to the page origin when run.py serves it), else
// http://localhost:7747 (the render-only default, as in the public release).

let defaultViewMatrix = [-1,0,0,0,
    0,-1,0,0,
    0,0,1,0,
    0,0,0,1];

let yaw = 0;   // Rotation around the Y-axis
let pitch = 0; // Rotation around the X-axis
let movement =  [0, 0, 0]; // Movement vector initialized to 0,0,0
let trajectoryPointCount = 0;  // Only the count is tracked, for display
let viewMatrix = defaultViewMatrix;
let socket = null;
let active_camera = null;

// Base focal length of the generation camera. The server sends
// init_focal_length in 'server-config'; zoom-out (B) never goes below it and
// run.py classifies a generation as zoom-in when fx > init_focal_length.
const DEFAULT_FOCAL_LENGTH = 1024;
let baseFocalLength = DEFAULT_FOCAL_LENGTH;
// Optional features reported by the server ('server-config').
let serverFeatures = {};

// One B press: focal length / 1.05, floored at the base. V x n then B x n can
// leave rounding error just above the base (e.g. 800.0000000000001), which
// run.py would classify as a zoom-in, so snap to the base within 1e-6.
function zoomOutFocal(f) {
    const out = f / 1.05;
    return out < baseFocalLength * (1 + 1e-6) ? baseFocalLength : out;
}

const canvas = document.getElementById("canvas");
const ctx = canvas ? canvas.getContext('2d') : null;
const serverConnect = document.getElementById("server-connect");
const fps = document.getElementById("fps");
const iter_number = document.getElementById("iter-number");
const focal_x = document.getElementById("focal-x");
const focal_y = document.getElementById("focal-y");
const send_button = document.getElementById("send-button");
const prompt_box = document.getElementById("prompt-box");
const status_line = document.getElementById("status-line");
const session_dir_el = document.getElementById("session-dir");
const scene_stats_el = document.getElementById("scene-stats");
const current_prompt_el = document.getElementById("current-prompt");
const message_el = document.getElementById("message");

// Null-safe setters for optional UI elements
function safeSetText(el, text) { if (el) el.innerText = text; }
function safeSetValue(el, val) { if (el) el.value = val; }

// Emit only when a socket exists (keys may be pressed before connecting).
function emitToServer(event, ...args) {
    if (socket) socket.emit(event, ...args);
}

function resolveServerUrl() {
    let param = null;
    try {
        param = new URLSearchParams(window.location.search).get('server');
    } catch (e) {
        param = null;
    }
    if (param) {
        param = param.trim();
        if (!/^[a-z][a-z0-9+.-]*:\/\//i.test(param)) param = 'http://' + param;
        return param;
    }
    // Only the page that run.py serves (index_gen.html) sets this. The
    // render-only page keeps connecting to localhost:7747 even when some other
    // static server hosts it, because run_render_only.py serves no pages.
    if (typeof window.WZ_DEFAULT_SERVER === 'string' && window.WZ_DEFAULT_SERVER) {
        return window.WZ_DEFAULT_SERVER;
    }
    return 'http://localhost:7747';
}

const cameras = [
    {
        id: 0,
        position: [
            0, 0, 0   // +left, +up, +forward
        ],
        rotation: [
            [-1, 0, 0],
            [0., -1, 0],
            [0, 0, 1],
        ],
        fy: DEFAULT_FOCAL_LENGTH,
        fx: DEFAULT_FOCAL_LENGTH,
        yaw: 0,
        pitch: 0,
        movement: [0, 0, 0],
    },
];

function getViewMatrix(camera) {
    const R = camera.rotation.flat();
    const t = camera.position;
    const camToWorld = [
        [R[0], R[1], R[2], 0],
        [R[3], R[4], R[5], 0],
        [R[6], R[7], R[8], 0],
        [
            -t[0] * R[0] - t[1] * R[3] - t[2] * R[6],
            -t[0] * R[1] - t[1] * R[4] - t[2] * R[7],
            -t[0] * R[2] - t[1] * R[5] - t[2] * R[8],
            1,
        ],
    ].flat();
    return camToWorld;
}

function multiply4(a, b) {
    return [
        b[0] * a[0] + b[1] * a[4] + b[2] * a[8] + b[3] * a[12],
        b[0] * a[1] + b[1] * a[5] + b[2] * a[9] + b[3] * a[13],
        b[0] * a[2] + b[1] * a[6] + b[2] * a[10] + b[3] * a[14],
        b[0] * a[3] + b[1] * a[7] + b[2] * a[11] + b[3] * a[15],
        b[4] * a[0] + b[5] * a[4] + b[6] * a[8] + b[7] * a[12],
        b[4] * a[1] + b[5] * a[5] + b[6] * a[9] + b[7] * a[13],
        b[4] * a[2] + b[5] * a[6] + b[6] * a[10] + b[7] * a[14],
        b[4] * a[3] + b[5] * a[7] + b[6] * a[11] + b[7] * a[15],
        b[8] * a[0] + b[9] * a[4] + b[10] * a[8] + b[11] * a[12],
        b[8] * a[1] + b[9] * a[5] + b[10] * a[9] + b[11] * a[13],
        b[8] * a[2] + b[9] * a[6] + b[10] * a[10] + b[11] * a[14],
        b[8] * a[3] + b[9] * a[7] + b[10] * a[11] + b[11] * a[15],
        b[12] * a[0] + b[13] * a[4] + b[14] * a[8] + b[15] * a[12],
        b[12] * a[1] + b[13] * a[5] + b[14] * a[9] + b[15] * a[13],
        b[12] * a[2] + b[13] * a[6] + b[14] * a[10] + b[15] * a[14],
        b[12] * a[3] + b[13] * a[7] + b[14] * a[11] + b[15] * a[15],
    ];
}

function invert4(a) {
    let b00 = a[0] * a[5] - a[1] * a[4];
    let b01 = a[0] * a[6] - a[2] * a[4];
    let b02 = a[0] * a[7] - a[3] * a[4];
    let b03 = a[1] * a[6] - a[2] * a[5];
    let b04 = a[1] * a[7] - a[3] * a[5];
    let b05 = a[2] * a[7] - a[3] * a[6];
    let b06 = a[8] * a[13] - a[9] * a[12];
    let b07 = a[8] * a[14] - a[10] * a[12];
    let b08 = a[8] * a[15] - a[11] * a[12];
    let b09 = a[9] * a[14] - a[10] * a[13];
    let b10 = a[9] * a[15] - a[11] * a[13];
    let b11 = a[10] * a[15] - a[11] * a[14];
    let det =
        b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (!det) return null;
    return [
        (a[5] * b11 - a[6] * b10 + a[7] * b09) / det,
        (a[2] * b10 - a[1] * b11 - a[3] * b09) / det,
        (a[13] * b05 - a[14] * b04 + a[15] * b03) / det,
        (a[10] * b04 - a[9] * b05 - a[11] * b03) / det,
        (a[6] * b08 - a[4] * b11 - a[7] * b07) / det,
        (a[0] * b11 - a[2] * b08 + a[3] * b07) / det,
        (a[14] * b02 - a[12] * b05 - a[15] * b01) / det,
        (a[8] * b05 - a[10] * b02 + a[11] * b01) / det,
        (a[4] * b10 - a[5] * b08 + a[7] * b06) / det,
        (a[1] * b08 - a[0] * b10 - a[3] * b06) / det,
        (a[12] * b04 - a[13] * b02 + a[15] * b00) / det,
        (a[9] * b02 - a[8] * b04 - a[11] * b00) / det,
        (a[5] * b07 - a[4] * b09 - a[6] * b06) / det,
        (a[0] * b09 - a[1] * b07 + a[2] * b06) / det,
        (a[13] * b01 - a[12] * b03 - a[14] * b00) / det,
        (a[8] * b03 - a[9] * b01 + a[10] * b00) / det,
    ];
}

function rotate4(a, rad, x, y, z) {
    let len = Math.hypot(x, y, z);
    x /= len;
    y /= len;
    z /= len;
    let s = Math.sin(rad);
    let c = Math.cos(rad);
    let t = 1 - c;
    let b00 = x * x * t + c;
    let b01 = y * x * t + z * s;
    let b02 = z * x * t - y * s;
    let b10 = x * y * t - z * s;
    let b11 = y * y * t + c;
    let b12 = z * y * t + x * s;
    let b20 = x * z * t + y * s;
    let b21 = y * z * t - x * s;
    let b22 = z * z * t + c;
    return [
        a[0] * b00 + a[4] * b01 + a[8] * b02,
        a[1] * b00 + a[5] * b01 + a[9] * b02,
        a[2] * b00 + a[6] * b01 + a[10] * b02,
        a[3] * b00 + a[7] * b01 + a[11] * b02,
        a[0] * b10 + a[4] * b11 + a[8] * b12,
        a[1] * b10 + a[5] * b11 + a[9] * b12,
        a[2] * b10 + a[6] * b11 + a[10] * b12,
        a[3] * b10 + a[7] * b11 + a[11] * b12,
        a[0] * b20 + a[4] * b21 + a[8] * b22,
        a[1] * b20 + a[5] * b21 + a[9] * b22,
        a[2] * b20 + a[6] * b21 + a[10] * b22,
        a[3] * b20 + a[7] * b21 + a[11] * b22,
        ...a.slice(12, 16),
    ];
}

function translate4(a, x, y, z) {
    return [
        ...a.slice(0, 12),
        a[0] * x + a[4] * y + a[8] * z + a[12],
        a[1] * x + a[5] * y + a[9] * z + a[13],
        a[2] * x + a[6] * y + a[10] * z + a[14],
        a[3] * x + a[7] * y + a[11] * z + a[15],
    ];
}

// View matrix of the current camera (yaw, pitch, movement).
function currentViewMatrix() {
    let inv = invert4(defaultViewMatrix);
    inv = translate4(inv, ...movement);
    inv = rotate4(inv, yaw, 0, 1, 0);   // Yaw around the Y-axis
    inv = rotate4(inv, pitch, 1, 0, 0); // Pitch around the X-axis
    return invert4(inv);
}

const update_displayed_info = (camera) => {
    if (!camera) return;
    safeSetText(focal_x, Number(camera.fx).toFixed(1));
    safeSetText(focal_y, Number(camera.fy).toFixed(1));
};

// Show or dim UI parts that depend on optional server features.
// Elements with data-feature="<name>" get the class 'feature-off' when the
// feature is disabled; elements with data-feature-show="<name>" are hidden.
function applyFeatures(features) {
    serverFeatures = features || {};
    document.querySelectorAll('[data-feature]').forEach((el) => {
        const on = !!serverFeatures[el.getAttribute('data-feature')];
        el.classList.toggle('feature-off', !on);
    });
    document.querySelectorAll('[data-feature-show]').forEach((el) => {
        const on = !!serverFeatures[el.getAttribute('data-feature-show')];
        el.style.display = on ? '' : 'none';
    });
    if (prompt_box) {
        prompt_box.disabled = !serverFeatures.objects;
        prompt_box.placeholder = serverFeatures.objects
            ? 'e.g. a ladybug'
            : 'object insertion is disabled on this server';
    }
    if (send_button) send_button.disabled = !serverFeatures.objects;
}

function applyServerConfig(cfg) {
    if (!cfg) return;
    const f = Number(cfg.init_focal_length);
    if (isFinite(f) && f > 0) {
        const oldBase = baseFocalLength;
        baseFocalLength = f;
        cameras[0].fx = f;
        cameras[0].fy = f;
        if (active_camera) {
            if (active_camera.fx === oldBase && active_camera.fy === oldBase) {
                // Not zoomed yet: start at the server's base focal length.
                active_camera.fx = f;
                active_camera.fy = f;
            } else {
                active_camera.fx = Math.max(active_camera.fx, f);
                active_camera.fy = Math.max(active_camera.fy, f);
            }
            update_displayed_info(active_camera);
        }
    }
    // Keep the canvas aspect equal to the generation resolution.
    const h = Number(cfg.gen_H), w = Number(cfg.gen_W);
    if (canvas && isFinite(h) && isFinite(w) && h > 0 && w > 0) {
        canvas.height = Math.round(canvas.width * h / w);
    }
    applyFeatures(cfg.features);
}

// The trajectory lives on the server: trajectoryPointCount only mirrors the
// count that run.py reports. H and J show nothing locally; the server replies
// with 'server-state' "Trajectory point N added" / "Trajectory cleared", or
// "<event> ignored: <reason>" when it refuses (that text is shown as is).
// run_render_only.py does not reply, so H changes nothing there.
function serverStateText(msg) {
    const text = (msg === null || msg === undefined) ? '' : String(msg);
    const added = /^Trajectory point (\d+) added$/.exec(text);
    if (added) {
        trajectoryPointCount = parseInt(added[1], 10);
        return `Trajectory point ${trajectoryPointCount} added. Press R to generate video.`;
    }
    if (text === 'Trajectory cleared') {
        trajectoryPointCount = 0;
        return 'Trajectory cleared. Press H to add points.';
    }
    return text;
}

function applyServerStatus(st) {
    if (!st) return;
    // An accepted zoom-in or camera move uses up the trajectory: run.py clears
    // it when the job ends, also when the job fails.
    if (st.state === 'busy' && (st.job === 'zoom' || st.job === 'move')) trajectoryPointCount = 0;
    const state = st.state || 'unknown';
    let text = state;
    if (st.job) text += ' [' + st.job + ']';
    if (st.message) text += ': ' + st.message;
    safeSetText(status_line, text);
    if (status_line) status_line.setAttribute('data-state', state);
    if (st.session_dir) safeSetText(session_dir_el, st.session_dir);
    document.body.setAttribute('data-server-state', state);
}

function applySceneStats(stats) {
    if (!stats || !scene_stats_el) return;
    const parts = [];
    if (stats.num_points !== undefined && stats.num_points !== null) {
        parts.push(Number(stats.num_points).toLocaleString() + ' points');
    }
    if (Array.isArray(stats.labels) && stats.labels.length) {
        parts.push('labels: ' + stats.labels.join(', '));
    }
    if (stats.last_job) {
        let s = 'last job: ' + stats.last_job;
        if (typeof stats.seconds === 'number') s += ' (' + stats.seconds.toFixed(1) + ' s)';
        parts.push(s);
    }
    safeSetText(scene_stats_el, parts.join('  |  '));
}

// Attach an mp4 stream event to an optional <video> element.
function bindVideo(event, elementId) {
    const el = document.getElementById(elementId);
    let url = null;
    socket.on(event, (data) => {
        if (!el) return;
        if (url) URL.revokeObjectURL(url);
        const blob = new Blob([data], { type: 'video/mp4' });
        url = URL.createObjectURL(blob);
        el.src = url;
        const p = el.play();
        if (p && p.catch) p.catch(() => {});
    });
}

function connectToServer() {
    const serverUrl = resolveServerUrl();
    safeSetText(serverConnect, "Connecting to " + serverUrl + " ...");
    socket = io.connect(serverUrl);

    socket.on('connect', () => {
        console.log("Connected to server.");
        safeSetText(serverConnect, "Connected to server.");
    });

    socket.on('connect_error', () => {
        console.log("Connection failed.");
        safeSetText(serverConnect, "Connection to server failed. Please retry.");
    });

    socket.on('disconnect', () => {
        safeSetText(serverConnect, "Disconnected from server.");
        applyServerStatus({ state: 'error', message: 'disconnected' });
    });

    socket.on('frame', (data) => {
        if (!ctx) return;
        // Receive the rendered image data from the server
        const blob = new Blob([data], { type: 'image/jpeg' });
        const imageURL = URL.createObjectURL(blob);

        // Update the canvas with the received image
        const img = new Image();
        img.onload = () => {
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            URL.revokeObjectURL(imageURL);
        };
        img.src = imageURL;
    });

    socket.on('server-state', (msg) => {
        console.log(msg);
        safeSetText(serverConnect, serverStateText(msg));
    });

    socket.on('server-config', applyServerConfig);
    socket.on('server-status', applyServerStatus);
    socket.on('scene-stats', applySceneStats);

    socket.on('iter-number', (msg) => {
        safeSetText(iter_number, msg);
    });

    socket.on('scene-prompt', (msg) => {
        console.log(msg);
        const text = (msg === null || msg === undefined) ? '' : String(msg);
        if (prompt_box && document.activeElement !== prompt_box) safeSetValue(prompt_box, text);
        safeSetText(current_prompt_el, text ? text : '(none)');
    });

    bindVideo('rough-video', 'roughVideo');
    bindVideo('out-video', 'outVideo');
    bindVideo('concat-video', 'concatVideo');
}

function sendCameraPose() {
    if (socket && socket.connected && active_camera) {
        socket.emit('render-pose', {
            viewMatrix: viewMatrix,
            fx: active_camera.fx,
            fy: active_camera.fy
        });
    }
}

function sendScenePrompt() {
    if (!prompt_box) return;
    const text = prompt_box.value.trim();
    emitToServer('scene-prompt', text);
    safeSetText(current_prompt_el, text ? text : '(none)');
    // The server keeps the prompt until a zoom-in uses it (camera moves leave
    // it in place) and then clears it; 'current:' shows the server's copy.
    safeSetText(serverConnect, text
        ? `Object "${text}" will be inserted at the end of the next zoom-in (camera moves keep it).`
        : "Object prompt cleared.");
    // Give the focus back to the page so the key bindings work again.
    prompt_box.blur();
    if (send_button) send_button.blur();
}

// Keys must not fire while the user is typing into a text field.
function isTypingTarget(el) {
    if (!el || el === document.body) return false;
    const tag = (el.tagName || '').toUpperCase();
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable;
}

// Main function
async function main() {
    active_camera = JSON.parse(JSON.stringify(cameras[0]));  // deep copy
    update_displayed_info(active_camera);
    connectToServer();

    if (send_button) {
        send_button.addEventListener("click", sendScenePrompt);
    }
    if (prompt_box) {
        prompt_box.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                sendScenePrompt();
            } else if (e.key === "Escape") {
                prompt_box.blur();
            }
        });
    }

    let activeKeys = [];
    window.addEventListener("keydown", (e) => {
        if (isTypingTarget(document.activeElement)) return;

        // Browser shortcuts (Ctrl/Cmd+C, Ctrl+R, Ctrl+Z, ...) must not drive the
        // scene. With Ctrl, Cmd or Alt held only the documented Space combos
        // below are handled.
        const withModifier = e.ctrlKey || e.metaKey || e.altKey;

        if (e.code === "KeyV" && !withModifier) {  // Zoom in: focal length +5%
            active_camera.fx = Math.min(active_camera.fx * 1.05, 9999999);
            active_camera.fy = Math.min(active_camera.fy * 1.05, 9999999);
            update_displayed_info(active_camera);
        }
        if (e.code === "KeyB" && !withModifier) {  // Zoom out, never below the base focal length
            active_camera.fx = zoomOutFocal(active_camera.fx);
            active_camera.fy = zoomOutFocal(active_camera.fy);
            update_displayed_info(active_camera);
        }

        // One-shot commands below: ignore key auto-repeat and modifier combos.
        if (!e.repeat && !withModifier) {
            if (e.code === "KeyR") {
                // Add the current pose to the trajectory and generate
                viewMatrix = currentViewMatrix();
                emitToServer('gen', {
                    viewMatrix: viewMatrix,
                    fx: active_camera.fx,
                    fy: active_camera.fy,
                    addToTrajectory: true
                });
                console.log(`Generating video with trajectory (${trajectoryPointCount + 1} points)`);
                // A refusal ("... ignored: <reason>") replaces this text; an
                // accepted request resets the count (applyServerStatus).
                safeSetText(serverConnect, `Generating video with trajectory...`);
            }

            if (e.code === "KeyQ") {  // Toggle rewrite / overwrite for zoom-in
                emitToServer('rewrite');
            }

            if (e.code === "KeyJ") {
                // Count and message follow the server's reply (serverStateText).
                emitToServer('clear-trajectory');
                console.log("Clear trajectory requested");
            }

            if (e.code === "KeyH") {
                viewMatrix = currentViewMatrix();
                // Count and message follow the server's reply (serverStateText).
                emitToServer('add-trajectory-point', {
                    viewMatrix: viewMatrix,
                    fx: active_camera.fx,
                    fy: active_camera.fy
                });
                console.log("Trajectory point requested");
            }

            if (e.code === "KeyP" && serverFeatures.objects) {
                emitToServer('complete-background');
            }

            if (e.code === "KeyZ") {  // Undo the last generation
                emitToServer('undo');
            }

            if (e.code === "KeyX") {  // Save the scene
                emitToServer('save');
            }

            if (e.code === "KeyC") {  // Delete the Gaussians visible in this view
                viewMatrix = currentViewMatrix();
                emitToServer('delete', viewMatrix);
            }
        }

        if (e.code === 'Space') {
            e.preventDefault();
            if (!e.repeat) {
                if (e.ctrlKey && e.shiftKey) {
                    // Ctrl+Shift+Space: high-quality novel views (Gen3C)
                    emitToServer('generate-nvs-hq');
                    console.log("nvs-hq");
                } else if (e.ctrlKey && e.altKey) {
                    // Ctrl+Alt+Space: fix small cracks
                    emitToServer('fix-small-cracks');
                    console.log("fix-small-cracks");
                } else if (!withModifier) {
                    // Space: orbit preview
                    emitToServer('generate-nvs');
                    console.log("nvs");
                }
            }
        }

        // Held movement keys. Skipped under Ctrl/Cmd/Alt: browser shortcuts
        // such as Cmd+A or Ctrl+D may never deliver the matching keyup.
        if (!withModifier && !activeKeys.includes(e.code)) activeKeys.push(e.code);
    });

    window.addEventListener("keyup", (e) => {
        activeKeys = activeKeys.filter((k) => k !== e.code);
    });

    window.addEventListener("blur", () => {
        activeKeys = [];
    });

    let lastFrame = 0;
    let avgFps = 0;

    const frame = (now) => {
        let inv = invert4(defaultViewMatrix);
        let speed_factor = 0.2 * (1024 / active_camera.fx);

        if (activeKeys.includes("KeyA")) yaw -= 0.02 * speed_factor;
        if (activeKeys.includes("KeyD")) yaw += 0.02 * speed_factor;
        if (activeKeys.includes("KeyW")) pitch += 0.005 * speed_factor;
        if (activeKeys.includes("KeyS")) pitch -= 0.005 * speed_factor;

        pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, pitch));

        // Compute movement vector increment based on yaw
        let dx = 0, dz = 0, dy = 0;
        speed_factor = 1.0 * Math.pow((1024 / active_camera.fx), 0.25);
        if (activeKeys.includes("ArrowUp")) dz += 0.02 * speed_factor;
        if (activeKeys.includes("ArrowDown")) dz -= 0.02 * speed_factor;
        if (activeKeys.includes("ArrowLeft")) dx -= 0.02 * speed_factor;
        if (activeKeys.includes("ArrowRight")) dx += 0.02 * speed_factor;
        if (activeKeys.includes("KeyN")) dy -= 0.02 * speed_factor;
        if (activeKeys.includes("KeyM")) dy += 0.02 * speed_factor;

        // Convert dx and dz into world coordinates based on yaw
        let forward = [Math.sin(yaw) * dz, 0, Math.cos(yaw) * dz];
        let right = [Math.sin(yaw + Math.PI / 2) * dx, 0, Math.cos(yaw + Math.PI / 2) * dx];

        // Update movement vector
        movement[0] += forward[0] + right[0];
        movement[1] += forward[1] + right[1] + dy; // This should generally remain 0 in a FPS
        movement[2] += forward[2] + right[2];

        // Apply translation based on movement vector
        inv = translate4(inv, ...movement);

        // Apply rotations
        inv = rotate4(inv, yaw, 0, 1, 0); // Yaw around the Y-axis
        inv = rotate4(inv, pitch, 1, 0, 0); // Pitch around the X-axis

        // Compute the view matrix
        viewMatrix = invert4(inv);

        const currentFps = 1000 / (now - lastFrame) || 0;
        avgFps = avgFps * 0.9 + currentFps * 0.1;

        safeSetText(fps, Math.round(avgFps) + " fps");
        lastFrame = now;
        requestAnimationFrame(frame);
    };

    frame();

    // Send camera pose updates to the server at 60 Hz
    setInterval(sendCameraPose, 1000 / 60);
}

main().catch((err) => {
    console.error(err);
    safeSetText(message_el || serverConnect, err.toString());
});
