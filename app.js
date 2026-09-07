// Main Application UI coordinator for UVR5 Browser Edition

const translations = {
    en: {
        title: "Ultimate Vocal Remover - Web Edition",
        subtitle: "Client-side Vocal & Instrumental separation running entirely in your browser",
        choose_file: "Select Audio File",
        choose_file_hint: "or drag & drop file here",
        model: "Choose Model",
        convert: "Start Processing",
        stop: "Stop",
        download_vocals: "Download Vocals",
        download_instruments: "Download Instruments",
        progress_download: "Download progress",
        progress_processing: "Separation progress",
        console_log: "Processing Log",
        lang_label: "עברית",
        status_idle: "Ready",
        select_model: "Select Model...",
        format_label: "Save Format",
        audio_tools: "Audio Tools",
        pitch_shift: "Pitch Shift (semitones)",
        time_stretch: "Time Stretch (factor)",
        align_label: "Align reference and target tracks",
        align_btn: "Run Alignment",
        match_btn: "Match Volume (RMS)"
    },
    he: {
        title: "Ultimate Vocal Remover - גרסת דפדפן",
        subtitle: "הפרדת שירה וכלי נגינה בצד הלקוח הפועלת לחלוטין בדפדפן שלך",
        choose_file: "בחר קובץ שמע",
        choose_file_hint: "או גרור ושחרר קובץ כאן",
        model: "בחר מודל עיבוד",
        convert: "התחל עיבוד",
        stop: "עצור",
        download_vocals: "הורד שירה (Vocals)",
        download_instruments: "הורד כלי נגינה (Instruments)",
        progress_download: "התקדמות הורדת המודל",
        progress_processing: "התקדמות הפרדת האודיו",
        console_log: "לוג פעילות עיבוד",
        lang_label: "English",
        status_idle: "מוכן לפעולה",
        select_model: "בחר מודל...",
        format_label: "פורמט שמירה",
        audio_tools: "כלי אודיו",
        pitch_shift: "שינוי גובה צליל (Pitch)",
        time_stretch: "מתיחת זמן (Time Stretch)",
        align_label: "יישור סנכרון בין רצועות",
        align_btn: "בצע יישור סנכרון",
        match_btn: "התאם עוצמת שמע (RMS)"
    }
};

let currentLang = "en";
let worker = null;
let audioContext = null;
let decodedAudio = null;
let selectedModelName = "UVR-DeNoise-Lite";

const modelsList = [
    {
        name: "UVR-DeNoise-Lite",
        url: "./converted_models/UVR-DeNoise-Lite-single.onnx",
        dataUrl: null
    }
];

function initWorker() {
    if (worker) worker.terminate();
    worker = new Worker("worker.js");

    worker.onmessage = function (e) {
        const { status, data } = e.data;
        if (status === "status") {
            log(data);
        } else if (status === "download-progress") {
            updateDownloadProgress(data.percent);
        } else if (status === "processing-progress") {
            updateProcessingProgress(data);
        } else if (status === "model-loaded") {
            document.getElementById("convert-btn").disabled = false;
        } else if (status === "result") {
            handleResult(data);
        } else if (status === "error") {
            log(`Error: ${data}`);
            alert(`An error occurred: ${data}`);
            resetUI();
        }
    };
}

function log(message) {
    const consoleBox = document.getElementById("console-log");
    consoleBox.value += `\n[${new Date().toLocaleTimeString()}] ${message}`;
    consoleBox.scrollTop = consoleBox.scrollHeight;
}

function updateDownloadProgress(percent) {
    const bar = document.getElementById("download-bar");
    bar.style.width = `${percent}%`;
    document.getElementById("download-percent").innerText = `${percent}%`;
}

function updateProcessingProgress(percent) {
    const bar = document.getElementById("processing-bar");
    bar.style.width = `${percent}%`;
    document.getElementById("processing-percent").innerText = `${percent}%`;
}

function switchLanguage(lang) {
    currentLang = lang;
    document.documentElement.lang = lang;
    document.documentElement.dir = lang === "he" ? "rtl" : "ltr";

    document.getElementById("app-title").innerText = translations[lang].title;
    document.getElementById("app-subtitle").innerText = translations[lang].subtitle;
    document.getElementById("file-select-label").innerText = translations[lang].choose_file;
    document.getElementById("drag-drop-text").innerText = translations[lang].choose_file_hint;
    document.getElementById("model-label").innerText = translations[lang].model;
    document.getElementById("convert-btn").innerText = translations[lang].convert;
    document.getElementById("stop-btn").innerText = translations[lang].stop;
    document.getElementById("download-vocals-btn").innerText = translations[lang].download_vocals;
    document.getElementById("download-inst-btn").innerText = translations[lang].download_instruments;
    document.getElementById("download-label").innerText = translations[lang].progress_download;
    document.getElementById("processing-label").innerText = translations[lang].progress_processing;
    document.getElementById("log-label").innerText = translations[lang].console_log;
    document.getElementById("lang-toggle-btn").innerText = translations[lang].lang_label;
    document.getElementById("format-label").innerText = translations[lang].format_label;
    document.getElementById("audio-tools-label").innerText = translations[lang].audio_tools;
    document.getElementById("pitch-shift-label").innerText = translations[lang].pitch_shift;
    document.getElementById("time-stretch-label").innerText = translations[lang].time_stretch;
    document.getElementById("align-label").innerText = translations[lang].align_label;
    document.getElementById("align-btn").innerText = translations[lang].align_btn;
    document.getElementById("match-btn").innerText = translations[lang].match_btn;
}

async function handleFile(file) {
    log(`Loading file: ${file.name} (${Math.round(file.size / 1024 / 1024 * 10) / 10} MB)...`);
    document.getElementById("file-name-display").innerText = file.name;

    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }

    try {
        const arrayBuffer = await file.arrayBuffer();
        log("Decoding audio data...");
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        const leftChannel = audioBuffer.getChannelData(0);
        const rightChannel = audioBuffer.numberOfChannels > 1 ? audioBuffer.getChannelData(1) : new Float32Array(leftChannel.length);

        decodedAudio = {
            leftChannel,
            rightChannel,
            sampleRate: audioBuffer.sampleRate,
            name: file.name.substring(0, file.name.lastIndexOf('.'))
        };
        log(`Successfully decoded ${audioBuffer.numberOfChannels} channels, sample rate: ${audioBuffer.sampleRate}Hz, length: ${Math.round(audioBuffer.duration)} seconds`);

        loadModel();
    } catch (err) {
        log(`Error decoding file: ${err.message}`);
        alert("Failed to decode audio file. Make sure it is a valid MP3, WAV, or FLAC.");
    }
}

function loadModel() {
    const model = modelsList.find(m => m.name === selectedModelName);
    if (!model) return;
    initWorker();
    worker.postMessage({
        action: "load-model",
        data: {
            name: model.name,
            url: model.url,
            dataUrl: model.dataUrl
        }
    });
}

function resetUI() {
    document.getElementById("convert-btn").disabled = false;
    document.getElementById("stop-btn").disabled = true;
    updateProcessingProgress(0);
}

function handleResult(data) {
    log("Reconstruction completed! Encoding to WAV files...");
    const { vocalLeft, vocalRight, instLeft, instRight } = data;
    const sampleRate = decodedAudio.sampleRate;

    const vocalBlob = writeWav(vocalLeft, vocalRight, sampleRate);
    const instBlob = writeWav(instLeft, instRight, sampleRate);

    const vocalUrl = URL.createObjectURL(vocalBlob);
    const instUrl = URL.createObjectURL(instBlob);

    const vBtn = document.getElementById("download-vocals-btn");
    vBtn.onclick = () => {
        const a = document.createElement("a");
        a.href = vocalUrl;
        a.download = `${decodedAudio.name}_(Vocals).wav`;
        a.click();
    };
    vBtn.disabled = false;

    const iBtn = document.getElementById("download-inst-btn");
    iBtn.onclick = () => {
        const a = document.createElement("a");
        a.href = instUrl;
        a.download = `${decodedAudio.name}_(Instrumental).wav`;
        a.click();
    };
    iBtn.disabled = false;

    log("Separation finished! Your files are ready to download.");
    resetUI();
}

function writeWav(left, right, sampleRate) {
    const buffer = new ArrayBuffer(44 + left.length * 2 * 2);
    const view = new DataView(buffer);

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + left.length * 2 * 2, true);
    writeString(view, 8, 'WAVE');

    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 2, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 4, true);
    view.setUint16(32, 4, true);
    view.setUint16(34, 16, true);

    writeString(view, 36, 'data');
    view.setUint32(40, left.length * 2 * 2, true);

    let index = 44;
    for (let i = 0; i < left.length; i++) {
        let valL = Math.max(-1, Math.min(1, left[i]));
        let valR = Math.max(-1, Math.min(1, right[i]));
        view.setInt16(index, valL < 0 ? valL * 0x8000 : valL * 0x7FFF, true);
        view.setInt16(index + 2, valR < 0 ? valR * 0x8000 : valR * 0x7FFF, true);
        index += 4;
    }

    return new Blob([view], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

window.onload = function () {
    switchLanguage("en");

    const select = document.getElementById("model-select");
    modelsList.forEach(m => {
        const opt = document.createElement("option");
        opt.value = m.name;
        opt.innerText = m.name;
        select.appendChild(opt);
    });

    select.addEventListener("change", function () {
        selectedModelName = this.value;
        loadModel();
    });

    const fileInput = document.getElementById("audio-file-input");
    fileInput.addEventListener("change", function (e) {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    const dropZone = document.getElementById("drag-drop-zone");
    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "#00adb5";
        dropZone.style.background = "#222831";
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.style.borderColor = "#393e46";
        dropZone.style.background = "transparent";
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "#393e46";
        dropZone.style.background = "transparent";
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    document.getElementById("convert-btn").addEventListener("click", function () {
        if (!decodedAudio) {
            alert("Please select an audio file first.");
            return;
        }
        this.disabled = true;
        document.getElementById("stop-btn").disabled = false;
        log("Sending audio buffers to Web Worker for processing...");

        const pitchVal = parseFloat(document.getElementById("pitch-shift-input").value) || 0;
        const stretchVal = parseFloat(document.getElementById("time-stretch-input").value) || 1.0;

        let left = decodedAudio.leftChannel;
        let right = decodedAudio.rightChannel;

        if (pitchVal !== 0 || stretchVal !== 1.0) {
            log("Applying pitch / time stretching audio pre-processing...");
            if (pitchVal !== 0) {
                left = pitchShift(left, pitchVal, decodedAudio.sampleRate);
                right = pitchShift(right, pitchVal, decodedAudio.sampleRate);
            }
            if (stretchVal !== 1.0) {
                left = timeStretch(left, stretchVal);
                right = timeStretch(right, stretchVal);
            }
        }

        worker.postMessage({
            action: "process-audio",
            data: {
                leftChannel: left,
                rightChannel: right,
                sampleRate: decodedAudio.sampleRate
            }
        });
    });

    document.getElementById("stop-btn").addEventListener("click", function () {
        if (worker) {
            log("Halt requested. Terminating current Web Worker...");
            initWorker();
            resetUI();
        }
    });

    document.getElementById("lang-toggle-btn").addEventListener("click", function () {
        switchLanguage(currentLang === "en" ? "he" : "en");
    });
};
