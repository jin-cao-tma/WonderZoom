let defaultViewMatrix = [-1,0,0,0,
    0,-1,0,0,
    0,0,1,0,
    0,0,0,1];

let yaw = 0;   // Rotation around the Y-axis
let pitch = 0; // Rotation around the X-axis
let movement =  [0, 0, 0]; // Movement vector initialized to 0,0,0
let trajectoryPointCount = 0;  // 只需要记录点的数量用于显示
let viewMatrix = defaultViewMatrix;
let socket;
let currentCameraIndex = 0;
let projectionMatrix;
let active_camera = null;       // 先占位
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext('2d');
// const canvas_viz = document.getElementById("canvas-viz");
// const ctx_viz = canvas_viz.getContext('2d');
const serverConnect = document.getElementById("server-connect");
const fps = document.getElementById("fps");
const iter_number = document.getElementById("iter-number");
const camid = document.getElementById("camid");
const focal_x = document.getElementById("focal-x");
const focal_y = document.getElementById("focal-y");
const focal_length = document.getElementById("focal-length");
const inner_width = document.getElementById("inner-width");
const inner_height = document.getElementById("inner-height");
const send_button = document.getElementById("send-button");
const prompt_box = document.getElementById("prompt-box");
// Null-safe setter for optional UI elements
function safeSetText(el, text) { if (el) el.innerText = text; }
function safeSetValue(el, val) { if (el) el.value = val; }

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
        fy: 1024,
        fx: 1024,
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

const use_extrinsics = (camera) => {
    // viewMatrix = getViewMatrix(camera);
    yaw = camera.yaw;
    pitch = camera.pitch;
    movement = camera.movement;
    // defaultViewMatrix = viewMatrix;
};

const use_camera = (camera) => {
    // use_intrinsics(camera);
    use_extrinsics(camera);
};

const update_displayed_info = (camera) => {
    safeSetText(camid, "cam  " + currentCameraIndex);
    focal_x.innerText = "focal_x  " + camera.fx;
    focal_y.innerText = "focal_y  " + camera.fy;
    safeSetText(focal_length, "focal_length  " + Math.sqrt(camera.fx * camera.fy));
    safeSetText(inner_width, "inner_width  " + innerWidth);
    safeSetText(inner_height, "inner_height  " + innerHeight);
};

function connectToServer() {
    socket = io.connect('http://localhost:7747/');
    // socket = io.connect('http://10.79.12.218:7776/');
    // socket = io.connect('http://localhost:8000/');

    socket.on('connect', () => {
        console.log("Connected to server.");
        safeSetText(serverConnect, "Connected to server.");
    });

    socket.on('connect_error', () => {
        console.log("Connection failed.");
        safeSetText(serverConnect, "Connection to server failed. Please retry.");
    });
    
    socket.on('frame', (data) => {
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

    socket.on('viz', (data) => {
        // Receive the rendered image data from the server
        // const blob = new Blob([data], { type: 'image/jpeg' });
        // const imageURL = URL.createObjectURL(blob);

        // // Update the canvas with the received image
        // const img = new Image();
        // img.onload = () => {
        //     ctx_viz.drawImage(img, 0, 0, canvas_viz.width, canvas_viz.height);
        //     URL.revokeObjectURL(imageURL);
        // };
        // img.src = imageURL;
    });

    socket.on('server-state', (msg) => {
        console.log(msg);
        safeSetText(serverConnect, msg);
    });

    socket.on('iter-number', (msg) => {
        safeSetText(iter_number, msg);
    });

    socket.on('scene-prompt', (msg) => {
        console.log(msg);
        safeSetValue(prompt_box, msg);
    });

socket.on('goto-zoom-response', (data) => {
    if (data.success) {
        // console.log("Received zoom response:", data);
        
        // 1. 更新viewMatrix
        viewMatrix = data.viewMatrix;
        
        // 2. 更新相机参数
        active_camera.fx = data.focalLength;
        active_camera.fy = data.focalLength;
        
        // 3. 从viewMatrix更新全局状态变量
        const position = extractPositionFromViewMatrix(viewMatrix);
        const rotation = extractRotationFromViewMatrix(viewMatrix);
        
        // 更新movement
        movement = position;
        
        // 从旋转矩阵计算yaw和pitch
        const eulerAngles = extractEulerAnglesFromRotationMatrix(rotation);
        yaw = eulerAngles.yaw;
        pitch = eulerAngles.pitch;
        
        // 4. 更新显示
        update_displayed_info(active_camera);
        
        // 5. 发送新的相机位姿到服务器
        socket.emit('render-pose', {
            viewMatrix: viewMatrix,
            fx: active_camera.fx,
            fy: active_camera.fy
        });
        
        safeSetText(serverConnect, data.message);
    } else {
        console.log(`Failed to teleport: ${data.message}`);
        safeSetText(serverConnect, data.message);
    }
});

// 从旋转矩阵提取欧拉角


}
function extractEulerAnglesFromRotationMatrix(rotation) {
    // 从旋转矩阵中提取yaw和pitch
    const yaw = Math.atan2(rotation[0][2], rotation[2][2]);
    const pitch = Math.asin(-rotation[1][2]);
    
    return {
        yaw: yaw,
        pitch: Math.max(-Math.PI/2, Math.min(Math.PI/2, pitch))
    };
}

// 从view matrix提取位置
function extractPositionFromViewMatrix(matrix) {
    return [
        -matrix[12],
        -matrix[13],
        -matrix[14]
    ];
}

// 从view matrix提取旋转矩阵
function extractRotationFromViewMatrix(matrix) {
    return [
        [matrix[0], matrix[1], matrix[2]],
        [matrix[4], matrix[5], matrix[6]],
        [matrix[8], matrix[9], matrix[10]]
    ];
}function sendCameraPose() {
    if (socket && socket.connected) {
        // console.log("emit render-pose");
        socket.emit('render-pose', {
            viewMatrix: viewMatrix,
            fx: active_camera.fx,
            fy: active_camera.fy
        });
    }
}

function extractPositionFromViewMatrix(matrix) {
    return [matrix[12], matrix[13], -matrix[14]];
}

function extractRotationFromViewMatrix(matrix) {
    return [
        [matrix[0], matrix[1], matrix[2]],
        [matrix[4], matrix[5], matrix[6]],
        [matrix[8], matrix[9], matrix[10]]
    ];
}

function storeCameraPose(matrix, yaw, pitch, movement) {
    const newPosition = extractPositionFromViewMatrix(matrix);
    const newRotation = extractRotationFromViewMatrix(matrix);
    
    const camera_tmp = {
        id: cameras.length,
        position: newPosition,
        rotation: newRotation,
        fy: 1000,
        fx: 1000,
        yaw: yaw,
        pitch: pitch,
        movement: movement
    };
    console.log("camera_length: " + cameras.length);
    cameras.push(camera_tmp);
    if (cameras.length > 10) {
        console.log("camera_length exeeded: " + cameras.length);
        cameras.splice(1, 1);
    }
}

// Main function
async function main() {
    connectToServer();
    active_camera = JSON.parse(JSON.stringify(cameras[0]));  // deep copy
    update_displayed_info(active_camera);

    if (send_button) {
        send_button.addEventListener("click", () => {
            socket.emit('scene-prompt', prompt_box.value);
        });
    }
    let activeKeys = [];
    window.addEventListener("keydown", (e) => {
        if (document.activeElement != document.body) return;
        if (e.code === "KeyI") {
            socket.emit('start', 'start signal');  // Send start signal to the server
        }
        if (e.code === "KeyV") {  // Zoom in
            active_camera.fx =  Math.min(active_camera.fx * 1.05, 9999999);
              // Increase focal length by 5%
            active_camera.fy =  Math.min(active_camera.fy * 1.05, 9999999);
            update_displayed_info(active_camera);
            socket.emit('update_camera', active_camera);
        }
        if (e.code === "KeyB") {  // Zoom out
            active_camera.fx = Math.max(active_camera.fx / 1.05, 1024);
            active_camera.fy = Math.max(active_camera.fy / 1.05, 1024);
            update_displayed_info(active_camera);
            socket.emit('update_camera', active_camera);
        }

        if (e.code === "KeyR") {
            // 计算当前相机位姿
            let inv = invert4(defaultViewMatrix);
            // pitch = 0;
            inv = translate4(inv, ...movement);
            inv = rotate4(inv, yaw, 0, 1, 0);
            inv = rotate4(inv, pitch, 1, 0, 0);
            viewMatrix = invert4(inv);
            
            // 发送当前位姿和生成命令
            socket.emit('gen', {
                viewMatrix: viewMatrix,
                fx: active_camera.fx,
                fy: active_camera.fy,
                addToTrajectory: true  // 标识要先加到trajectory再生成
            });
            
            console.log(`Generating video with trajectory (${trajectoryPointCount + 1} points)`);
            safeSetText(serverConnect, `Generating video with trajectory...`);
            trajectoryPointCount = 0;  // 重置计数
        }
        

        if (e.code === "KeyQ") {
            socket.emit("rewrite")
            // let inv = invert4(defaultViewMatrix);
            // pitch = 0;
            // inv = translate4(inv, ...movement);

            // // Apply rotations
            // inv = rotate4(inv, yaw, 0, 1, 0); // Yaw around the Y-axis
            // inv = rotate4(inv, pitch, 1, 0, 0); // Pitch around the X-axis

            // // Compute the view matrix
            // viewMatrix = invert4(inv);

            // console.log("viewMatrix: [" + viewMatrix + "]");
        }
        if (e.code === "KeyJ") {
            socket.emit('clear-trajectory');
            trajectoryPointCount = 0;
            console.log("Trajectory cleared");
            safeSetText(serverConnect, "Trajectory cleared. Press H to add points.");
        }
        if (e.code === "KeyH") {
        // 计算当前相机位姿
        let inv = invert4(defaultViewMatrix);
        // pitch = 0;
        inv = translate4(inv, ...movement);
        inv = rotate4(inv, yaw, 0, 1, 0);
        inv = rotate4(inv, pitch, 1, 0, 0);
        viewMatrix = invert4(inv);
        
        // 发送轨迹点到后端存储
        socket.emit('add-trajectory-point', {
            viewMatrix: viewMatrix,
            fx: active_camera.fx,
            fy: active_camera.fy
        });
        
        trajectoryPointCount++;
        console.log(`Added trajectory point ${trajectoryPointCount}`);
        safeSetText(serverConnect, `Trajectory point ${trajectoryPointCount} added. Press R to generate video.`);
        }

        if (e.code === "KeyT") {
            let inv = invert4(defaultViewMatrix);
            pitch = 0;
            
            let backwardMovement = [0, 0, -0.8];
            let combinedMovement = [
                movement[0] + backwardMovement[0],
                movement[1] + backwardMovement[1],
                movement[2] + backwardMovement[2]
            ];

            inv = translate4(inv, ...combinedMovement);

            // Apply rotations
            inv = rotate4(inv, yaw, 0, 1, 0); // Yaw around the Y-axis
            inv = rotate4(inv, pitch, 1, 0, 0); // Pitch around the X-axis

            // Compute the view matrix
            viewMatrix = invert4(inv);
            let yaw_tmp = yaw;
            let pitch_tmp = pitch;
            let movement_tmp = combinedMovement;
            storeCameraPose(viewMatrix, yaw_tmp, pitch_tmp, movement_tmp);

            socket.emit('gen', viewMatrix);  // Send generate signal to the server
        }
        if (e.code === "KeyY") {
            let inv = invert4(defaultViewMatrix);
            pitch = 0;
            let leftTurnAngle = 20 * Math.PI / 180;

            inv = translate4(inv, ...movement);

            // Apply rotations
            inv = rotate4(inv, yaw - leftTurnAngle, 0, 1, 0); // Yaw around the Y-axis
            inv = rotate4(inv, pitch, 1, 0, 0); // Pitch around the X-axis

            // Compute the view matrix
            viewMatrix = invert4(inv);
            let yaw_tmp = yaw - leftTurnAngle;
            let pitch_tmp = pitch;
            let movement_tmp = movement;
            storeCameraPose(viewMatrix, yaw_tmp, pitch_tmp, movement_tmp);

            socket.emit('gen', viewMatrix);  // Send generate signal to the server
        }
        if (e.code === "KeyU") {
            let inv = invert4(defaultViewMatrix);
            pitch = 0;
            let rightTurnAngle = 20 * Math.PI / 180;

            inv = translate4(inv, ...movement);

            // Apply rotations
            inv = rotate4(inv, yaw + rightTurnAngle, 0, 1, 0); // Yaw around the Y-axis
            inv = rotate4(inv, pitch, 1, 0, 0); // Pitch around the X-axis

            // Compute the view matrix
            viewMatrix = invert4(inv);
            let yaw_tmp = yaw + rightTurnAngle;
            let pitch_tmp = pitch;
            let movement_tmp = movement;
            storeCameraPose(viewMatrix, yaw_tmp, pitch_tmp, movement_tmp);

            socket.emit('gen', viewMatrix);  // Send generate signal to the server
        }
        if (e.code === "KeyI") {
            let inv = invert4(defaultViewMatrix);
            pitch = 0;
            let leftTurnAngle = 15 * Math.PI / 180;

            let backwardMovement = [0, 0, -0.5];
            let combinedMovement = [
                movement[0] + backwardMovement[0],
                movement[1] + backwardMovement[1],
                movement[2] + backwardMovement[2]
            ];

            inv = translate4(inv, ...combinedMovement);

            // Apply rotations
            inv = rotate4(inv, yaw - leftTurnAngle, 0, 1, 0); // Yaw around the Y-axis
            inv = rotate4(inv, pitch, 1, 0, 0); // Pitch around the X-axis

            // Compute the view matrix
            viewMatrix = invert4(inv);
            let yaw_tmp = yaw - leftTurnAngle;
            let pitch_tmp = pitch;
            let movement_tmp = combinedMovement;
            storeCameraPose(viewMatrix, yaw_tmp, pitch_tmp, movement_tmp);

            socket.emit('gen', viewMatrix);  // Send generate signal to the server
        }
        if (e.code === "KeyP") {
        socket.emit('goto-nearest-zoom');
    }
        if (e.code === "KeyO") {
            let inv = invert4(defaultViewMatrix);
            pitch = 0;
            let rightTurnAngle = 15 * Math.PI / 180;

            let backwardMovement = [0, 0, -0.5];
            let combinedMovement = [
                movement[0] + backwardMovement[0],
                movement[1] + backwardMovement[1],
                movement[2] + backwardMovement[2]
            ];

            inv = translate4(inv, ...combinedMovement);

            // Apply rotations
            inv = rotate4(inv, yaw + rightTurnAngle, 0, 1, 0); // Yaw around the Y-axis
            inv = rotate4(inv, pitch, 1, 0, 0); // Pitch around the X-axis

            // Compute the view matrix
            viewMatrix = invert4(inv);
            let yaw_tmp = yaw + rightTurnAngle;
            let pitch_tmp = pitch;
            let movement_tmp = combinedMovement;
            storeCameraPose(viewMatrix, yaw_tmp, pitch_tmp, movement_tmp);
            
            socket.emit('gen', viewMatrix);  // Send generate signal to the server
        }
    if (e.code === 'Space') {
        if ( e.ctrlKey && e.shiftKey) {
            // Ctrl+Space: 高质量NVS
            e.preventDefault();
            socket.emit('generate-nvs-hq');
            console.log("nvs-hq");  // <-- 这行修正了
        } 
        else if (e.ctrlKey && e.altKey){
            e.preventDefault();
            socket.emit('fix-small-cracks');
            console.log("small-hq");
        }
        
        else {
            // Space: 普通NVS
            e.preventDefault();
            socket.emit('generate-nvs');
            console.log("nvs");  // 可选添加
        } 
    }               if (e.code === "KeyK") {
            let inv = invert4(defaultViewMatrix);
            pitch = 0;
            let leftTurnAngle = 15 * Math.PI / 180;

            let backwardMovement = [0, 0, 0.5];
            let combinedMovement = [
                movement[0] + backwardMovement[0],
                movement[1] + backwardMovement[1],
                movement[2] + backwardMovement[2]
            ];

            inv = translate4(inv, ...combinedMovement);

            // Apply rotations
            inv = rotate4(inv, yaw - leftTurnAngle, 0, 1, 0); // Yaw around the Y-axis
            inv = rotate4(inv, pitch, 1, 0, 0); // Pitch around the X-axis

            // Compute the view matrix
            viewMatrix = invert4(inv);
            let yaw_tmp = yaw - leftTurnAngle;
            let pitch_tmp = pitch;
            let movement_tmp = combinedMovement;
            storeCameraPose(viewMatrix, yaw_tmp, pitch_tmp, movement_tmp);

            socket.emit('gen', viewMatrix);  // Send generate signal to the server
        }
        if (e.code === "KeyL") {
            let inv = invert4(defaultViewMatrix);
            pitch = 0;
            let rightTurnAngle = 15 * Math.PI / 180;

            let backwardMovement = [0, 0, 0.5];
            let combinedMovement = [
                movement[0] + backwardMovement[0],
                movement[1] + backwardMovement[1],
                movement[2] + backwardMovement[2]
            ];

            inv = translate4(inv, ...combinedMovement);

            // Apply rotations
            inv = rotate4(inv, yaw + rightTurnAngle, 0, 1, 0); // Yaw around the Y-axis
            inv = rotate4(inv, pitch, 1, 0, 0); // Pitch around the X-axis

            // Compute the view matrix
            viewMatrix = invert4(inv);
            let yaw_tmp = yaw + rightTurnAngle;
            let pitch_tmp = pitch;
            let movement_tmp = combinedMovement;
            storeCameraPose(viewMatrix, yaw_tmp, pitch_tmp, movement_tmp);
            
            socket.emit('gen', viewMatrix);  // Send generate signal to the server
        }
        // if (e.code === "KeyF") {
        //     active_camera.fx += 10; // Adjust 10 to your desired increment value
        //     active_camera.fy += 10; // Adjust 10 to your desired increment value
        //     // use_intrinsics(active_camera);
        //     update_displayed_info(active_camera);
        // }

        // if (e.code === "KeyG") {
        //     active_camera.fx -= 10; // Adjust 10 to your desired decrement value
        //     active_camera.fy -= 10; // Adjust 10 to your desired decrement value
        //     // use_intrinsics(active_camera);
        //     update_displayed_info(active_camera);
        // }
        
        // Undo
        if (e.code === "KeyZ") {
            socket.emit('undo');
        }

        if (e.code === "KeyX") {
            socket.emit('save');
        }
        
        if (e.code === "KeyE") {
            socket.emit('fill_hole');
        }

        if (e.code === "KeyC") {
            let inv = invert4(defaultViewMatrix);
            inv = translate4(inv, ...movement);

            // Apply rotations
            inv = rotate4(inv, yaw, 0, 1, 0); // Yaw around the Y-axis
            inv = rotate4(inv, pitch, 1, 0, 0); // Pitch around the X-axis

            // Compute the view matrix
            viewMatrix = invert4(inv);

            socket.emit('delete', viewMatrix);
        }

        if (!activeKeys.includes(e.code)) activeKeys.push(e.code);
        if (/\d/.test(e.key)) {
            currentCameraIndex = parseInt(e.key)
            active_camera = JSON.parse(JSON.stringify(cameras[currentCameraIndex]));
            use_camera(active_camera);
            update_displayed_info(active_camera);
        }
    });

    window.addEventListener("keyup", (e) => {
        if (document.activeElement != document.body) return;
        activeKeys = activeKeys.filter((k) => k !== e.code);
    });

    window.addEventListener("blur", () => {
        activeKeys = [];
    });

    let lastFrame = 0;
    let avgFps = 0;

    const frame = (now) => {
        let inv = invert4(defaultViewMatrix);
        // speed_factor = 0.2;
        speed_factor = 0.2 * (1024 / active_camera.fx)
        // speed_factor = 0.01;
        
        if (activeKeys.includes("KeyA")) yaw -= 0.02 * speed_factor;
        if (activeKeys.includes("KeyD")) yaw += 0.02 * speed_factor;
        if (activeKeys.includes("KeyW")) pitch += 0.005 * speed_factor;
        if (activeKeys.includes("KeyS")) pitch -= 0.005 * speed_factor;

        pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, pitch));

        // Compute movement vector increment based on yaw
        let dx = 0, dz = 0, dy = 0;
        speed_factor = 1.0 * Math.pow((1024 / active_camera.fx),0.25)
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

        fps.innerText = Math.round(avgFps) + " fps";
        lastFrame = now;
        requestAnimationFrame(frame);
    };

    frame();

    // Send camera pose updates to the server every 50ms (20 FPS)
    setInterval(sendCameraPose, 1000 / 60);
}

main().catch((err) => {
    document.getElementById("message").innerText = err.toString();
});